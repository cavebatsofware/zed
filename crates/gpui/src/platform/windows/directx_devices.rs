use anyhow::{Context, Result};
use regex::Regex;
use util::ResultExt;
use windows::Win32::{
    Foundation::HMODULE,
    Graphics::{
        Direct3D::{
            D3D_DRIVER_TYPE_UNKNOWN, D3D_FEATURE_LEVEL, D3D_FEATURE_LEVEL_10_1,
            D3D_FEATURE_LEVEL_11_0, D3D_FEATURE_LEVEL_11_1,
        },
        Direct3D11::{
            D3D11_CREATE_DEVICE_BGRA_SUPPORT, D3D11_CREATE_DEVICE_DEBUG,
            D3D11_FEATURE_D3D10_X_HARDWARE_OPTIONS, D3D11_FEATURE_DATA_D3D10_X_HARDWARE_OPTIONS,
            D3D11_SDK_VERSION, D3D11CreateDevice, ID3D11Device, ID3D11DeviceContext,
        },
        Dxgi::{
            CreateDXGIFactory2, DXGI_ADAPTER_DESC1, DXGI_CREATE_FACTORY_DEBUG,
            DXGI_CREATE_FACTORY_FLAGS, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
            DXGI_GPU_PREFERENCE_MINIMUM_POWER, IDXGIAdapter1, IDXGIFactory6,
        },
    },
};

const INTEGRATED_GPU_MEMORY_THRESHOLD: u64 = 512 * 1024 * 1024;
const VENDOR_ID_NVIDIA: u32 = 0x10DE;
const VENDOR_ID_AMD: u32 = 0x1002;
const VENDOR_ID_INTEL: u32 = 0x8086;

#[derive(Debug, Clone, PartialEq)]
pub enum GpuPreference {
    Auto,
    HighPerformance,
    PowerEfficient,
    Specific(String),
}

impl Default for GpuPreference {
    fn default() -> Self {
        Self::Auto
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum GpuMobility {
    Desktop,
    Mobile,
    Integrated,
    Unknown,
}

#[derive(Debug, Clone, PartialEq)]
pub enum GpuVendor {
    Nvidia,
    Amd,
    Intel,
    Unknown(u32),
}

#[derive(Debug, Clone, PartialEq)]
pub enum GpuType {
    Dedicated { vendor: GpuVendor, memory_mb: u32 },
    Integrated { vendor: GpuVendor },
}

#[derive(Debug, Clone)]
pub struct GpuInfo {
    pub device_name: String,
    pub vendor_id: u32,
    pub device_id: u32,
    pub dedicated_memory: u64,
    pub shared_memory: u64,
}

#[derive(Debug, Clone)]
pub struct GpuCandidate {
    pub adapter: IDXGIAdapter1,
    pub info: GpuInfo,
    pub gpu_type: GpuType,
    pub mobility: GpuMobility,
    pub score: u32,
}

pub fn load_gpu_preference() -> GpuPreference {
    if let Ok(pref) = std::env::var("GPUI_GPU_PREFERENCE") {
        match pref.to_lowercase().as_str() {
            "high_performance" | "performance" | "dedicated" => GpuPreference::HighPerformance,
            "power_efficient" | "efficiency" | "integrated" | "power" => {
                GpuPreference::PowerEfficient
            }
            gpu_name => GpuPreference::Specific(gpu_name.to_string()),
        }
    } else {
        GpuPreference::Auto
    }
}

pub(crate) fn try_to_recover_from_device_lost<T>(
    mut f: impl FnMut() -> Result<T>,
    on_success: impl FnOnce(T),
    on_error: impl FnOnce(),
) {
    let result = (0..5).find_map(|i| {
        if i > 0 {
            // Add a small delay before retrying
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
        f().log_err()
    });

    if let Some(result) = result {
        on_success(result);
    } else {
        on_error();
    }
}

#[derive(Clone)]
pub(crate) struct DirectXDevices {
    pub(crate) adapter: IDXGIAdapter1,
    pub(crate) dxgi_factory: IDXGIFactory6,
    pub(crate) device: ID3D11Device,
    pub(crate) device_context: ID3D11DeviceContext,
}

impl DirectXDevices {
    pub(crate) fn new() -> Result<Self> {
        let debug_layer_available = check_debug_layer_available();
        let dxgi_factory =
            get_dxgi_factory(debug_layer_available).context("Creating DXGI factory")?;
        let adapter =
            get_adapter(&dxgi_factory, debug_layer_available).context("Getting DXGI adapter")?;
        let (device, device_context) = {
            let mut context: Option<ID3D11DeviceContext> = None;
            let mut feature_level = D3D_FEATURE_LEVEL::default();
            let device = get_device(
                &adapter,
                Some(&mut context),
                Some(&mut feature_level),
                debug_layer_available,
            )
            .context("Creating Direct3D device")?;
            match feature_level {
                D3D_FEATURE_LEVEL_11_1 => {
                    log::info!("Created device with Direct3D 11.1 feature level.")
                }
                D3D_FEATURE_LEVEL_11_0 => {
                    log::info!("Created device with Direct3D 11.0 feature level.")
                }
                D3D_FEATURE_LEVEL_10_1 => {
                    log::info!("Created device with Direct3D 10.1 feature level.")
                }
                _ => unreachable!(),
            }
            (
                device,
                context.ok_or_else(|| anyhow::anyhow!("Failed to create device context"))?,
            )
        };

        Ok(Self {
            adapter,
            dxgi_factory,
            device,
            device_context,
        })
    }
}

#[inline]
fn check_debug_layer_available() -> bool {
    #[cfg(debug_assertions)]
    {
        use windows::Win32::Graphics::Dxgi::{DXGIGetDebugInterface1, IDXGIInfoQueue};

        unsafe { DXGIGetDebugInterface1::<IDXGIInfoQueue>(0) }
            .log_err()
            .is_some()
    }
    #[cfg(not(debug_assertions))]
    {
        false
    }
}

#[inline]
fn get_dxgi_factory(debug_layer_available: bool) -> Result<IDXGIFactory6> {
    let factory_flag = if debug_layer_available {
        DXGI_CREATE_FACTORY_DEBUG
    } else {
        #[cfg(debug_assertions)]
        log::warn!(
            "Failed to get DXGI debug interface. DirectX debugging features will be disabled."
        );
        DXGI_CREATE_FACTORY_FLAGS::default()
    };
    unsafe { Ok(CreateDXGIFactory2(factory_flag)?) }
}

#[inline]
fn get_adapter(dxgi_factory: &IDXGIFactory6, debug_layer_available: bool) -> Result<IDXGIAdapter1> {
    let preference = load_gpu_preference();

    // Enumerate all GPU candidates
    let candidates = enumerate_gpu_candidates(dxgi_factory, debug_layer_available)
        .context("Failed to enumerate GPU candidates")?;

    if candidates.is_empty() {
        return Err(anyhow::anyhow!("No suitable GPUs found"));
    }

    // Log available GPUs
    log_available_gpus(&candidates);

    // Select GPU based on preference
    let selected =
        select_gpu_with_preference(&candidates, &preference).context("Failed to select GPU")?;

    log::info!(
        "Selected GPU: {} (Score: {}, Type: {:?}, Mobility: {:?})",
        selected.info.device_name,
        selected.score,
        selected.gpu_type,
        selected.mobility
    );

    Ok(selected.adapter.clone())
}

#[inline]
fn get_device(
    adapter: &IDXGIAdapter1,
    context: Option<*mut Option<ID3D11DeviceContext>>,
    feature_level: Option<*mut D3D_FEATURE_LEVEL>,
    debug_layer_available: bool,
) -> Result<ID3D11Device> {
    let mut device: Option<ID3D11Device> = None;
    let device_flags = if debug_layer_available {
        D3D11_CREATE_DEVICE_BGRA_SUPPORT | D3D11_CREATE_DEVICE_DEBUG
    } else {
        D3D11_CREATE_DEVICE_BGRA_SUPPORT
    };
    unsafe {
        D3D11CreateDevice(
            adapter,
            D3D_DRIVER_TYPE_UNKNOWN,
            HMODULE::default(),
            device_flags,
            // 4x MSAA is required for Direct3D Feature Level 10.1 or better
            Some(&[
                D3D_FEATURE_LEVEL_11_1,
                D3D_FEATURE_LEVEL_11_0,
                D3D_FEATURE_LEVEL_10_1,
            ]),
            D3D11_SDK_VERSION,
            Some(&mut device),
            feature_level,
            context,
        )?;
    }
    let device = device.unwrap();
    let mut data = D3D11_FEATURE_DATA_D3D10_X_HARDWARE_OPTIONS::default();
    unsafe {
        device
            .CheckFeatureSupport(
                D3D11_FEATURE_D3D10_X_HARDWARE_OPTIONS,
                &mut data as *mut _ as _,
                std::mem::size_of::<D3D11_FEATURE_DATA_D3D10_X_HARDWARE_OPTIONS>() as u32,
            )
            .context("Checking GPU device feature support")?;
    }
    if data
        .ComputeShaders_Plus_RawAndStructuredBuffers_Via_Shader_4_x
        .as_bool()
    {
        Ok(device)
    } else {
        Err(anyhow::anyhow!(
            "Required feature StructuredBuffer is not supported by GPU/driver"
        ))
    }
}

fn enumerate_gpu_candidates(
    dxgi_factory: &IDXGIFactory6,
    debug_layer_available: bool,
) -> Result<Vec<GpuCandidate>> {
    let mut candidates = Vec::new();
    let mut seen_adapters = std::collections::HashSet::new();

    // Try both performance preferences to enumerate all adapters
    for preference in [
        DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
        DXGI_GPU_PREFERENCE_MINIMUM_POWER,
    ] {
        for adapter_index in 0.. {
            let adapter: IDXGIAdapter1 =
                match unsafe { dxgi_factory.EnumAdapterByGpuPreference(adapter_index, preference) }
                {
                    Ok(adapter) => adapter,
                    Err(_) => break, // No more adapters for this preference
                };

            let desc = match unsafe { adapter.GetDesc1() } {
                Ok(desc) => desc,
                Err(_) => continue,
            };

            // Skip duplicates (same adapter enumerated with different preferences)
            let adapter_key = (desc.VendorId, desc.DeviceId, desc.SubSysId, desc.Revision);
            if seen_adapters.contains(&adapter_key) {
                continue;
            }
            seen_adapters.insert(adapter_key);

            // Check if adapter supports DirectX 11
            if get_device(&adapter, None, None, debug_layer_available)
                .log_err()
                .is_none()
            {
                continue;
            }

            let candidate = analyze_gpu_candidate(adapter, desc)?;
            candidates.push(candidate);
        }
    }

    // Sort by score (highest first)
    candidates.sort_by_key(|c| std::cmp::Reverse(c.score));

    Ok(candidates)
}

fn analyze_gpu_candidate(adapter: IDXGIAdapter1, desc: DXGI_ADAPTER_DESC1) -> Result<GpuCandidate> {
    let device_name = String::from_utf16_lossy(&desc.Description)
        .trim_matches(char::from(0))
        .to_string();

    let vendor = classify_vendor(desc.VendorId);
    let memory_mb = (desc.DedicatedVideoMemory / (1024 * 1024)) as u32;
    let is_integrated = desc.DedicatedVideoMemory < INTEGRATED_GPU_MEMORY_THRESHOLD;

    let gpu_type = if is_integrated {
        GpuType::Integrated { vendor }
    } else {
        GpuType::Dedicated { vendor, memory_mb }
    };

    let mobility = detect_gpu_mobility(&device_name, &desc);
    let score = calculate_gpu_score(&gpu_type, &mobility, &device_name);

    let info = GpuInfo {
        device_name,
        vendor_id: desc.VendorId,
        device_id: desc.DeviceId,
        dedicated_memory: desc.DedicatedVideoMemory as u64,
        shared_memory: desc.SharedSystemMemory as u64,
    };

    Ok(GpuCandidate {
        adapter,
        info,
        gpu_type,
        mobility,
        score,
    })
}

fn classify_vendor(vendor_id: u32) -> GpuVendor {
    match vendor_id {
        VENDOR_ID_NVIDIA => GpuVendor::Nvidia,
        VENDOR_ID_AMD => GpuVendor::Amd,
        VENDOR_ID_INTEL => GpuVendor::Intel,
        id => GpuVendor::Unknown(id),
    }
}

fn detect_gpu_mobility(device_name: &str, desc: &DXGI_ADAPTER_DESC1) -> GpuMobility {
    classify_gpu_by_explicit_markers(device_name)
        .or_else(|| classify_by_known_models(device_name))
        .unwrap_or_else(|| classify_by_advanced_heuristics(device_name, desc))
}

fn classify_gpu_by_explicit_markers(device_name: &str) -> Option<GpuMobility> {
    let name_lower = device_name.to_lowercase();

    let nvidia_mobile = Regex::new(r"\b(nvidia|geforce)\b.*\b(max-q|mobile|laptop)\b").ok()?;
    let amd_mobile_keyword = Regex::new(r"\b(amd|radeon)\b.*\bmobile\b").ok()?;
    let amd_mobile_suffix = Regex::new(r"\brx\s*\d{4}[ms]\b").ok()?;
    let intel_mobile = Regex::new(r"\bintel\b.*(iris\s+xe|uhd\s+graphics|arc\s+\w+m\b)").ok()?;

    if nvidia_mobile.is_match(&name_lower)
        || amd_mobile_keyword.is_match(&name_lower)
        || amd_mobile_suffix.is_match(&name_lower)
        || intel_mobile.is_match(&name_lower)
    {
        return Some(GpuMobility::Mobile);
    }

    None
}

fn classify_by_known_models(device_name: &str) -> Option<GpuMobility> {
    let name_lower = device_name.to_lowercase();

    let amd_known_mobile = Regex::new(r"\brx\s*(7[67]00s|7600m(\s+xt)?|6[6-8]00m)\b").ok()?;
    let intel_arc_mobile = Regex::new(r"\barc\s+a[3-7][3-7]0m\b").ok()?;
    let intel_arc_desktop = Regex::new(r"\barc\s+a(770|750|580|380|310)\b").ok()?;

    if amd_known_mobile.is_match(&name_lower) || intel_arc_mobile.is_match(&name_lower) {
        return Some(GpuMobility::Mobile);
    }

    if intel_arc_desktop.is_match(&name_lower) {
        return Some(GpuMobility::Desktop);
    }

    None
}

fn classify_by_advanced_heuristics(device_name: &str, desc: &DXGI_ADAPTER_DESC1) -> GpuMobility {
    let name_lower = device_name.to_lowercase();
    let vram_gb = desc.DedicatedVideoMemory / (1024 * 1024 * 1024);

    if desc.DedicatedVideoMemory < INTEGRATED_GPU_MEMORY_THRESHOLD
        || name_lower.contains("integrated")
    {
        return GpuMobility::Integrated;
    }

    if name_lower.contains("rtx") {
        if name_lower.contains("ti") && vram_gb >= 8 {
            return GpuMobility::Desktop;
        }
        if vram_gb >= 12 {
            return GpuMobility::Desktop;
        }
        if vram_gb <= 6 {
            return GpuMobility::Mobile;
        }
    }

    if let Ok(legacy_desktop) = Regex::new(r"\b(gtx\s+(750|760|96[08]|970|980)|rx\s+[4-5][7-8]0)\b")
    {
        if legacy_desktop.is_match(&name_lower) {
            return GpuMobility::Desktop;
        }
    }

    if vram_gb >= 6 {
        GpuMobility::Desktop
    } else {
        GpuMobility::Unknown
    }
}

fn calculate_gpu_score(gpu_type: &GpuType, mobility: &GpuMobility, device_name: &str) -> u32 {
    let mut score = 0u32;

    match gpu_type {
        GpuType::Dedicated { vendor, memory_mb } => {
            score += 1000; // Base score for dedicated GPU
            score += memory_mb / 10; // Memory bonus (100MB = 10 points)

            // Vendor bonuses (for gaming/graphics workloads)
            match vendor {
                GpuVendor::Nvidia => score += 200, // CUDA, better driver support
                GpuVendor::Amd => score += 150,
                GpuVendor::Intel => score += 100, // Arc GPUs
                _ => {}
            }

            // Performance tier detection
            let name_lower = device_name.to_lowercase();
            if name_lower.contains("rtx") || name_lower.contains("gtx") {
                score += 100;
            }
            if name_lower.contains("4090") || name_lower.contains("4080") {
                score += 500; // High-end bonus
            }

            // Mobility adjustment
            match mobility {
                GpuMobility::Desktop => score += 300, // Desktop preference for performance
                GpuMobility::Mobile => score -= 50,   // Slight mobile penalty
                _ => {}
            }
        }

        GpuType::Integrated { vendor } => {
            score += 100; // Base score for integrated
            match vendor {
                GpuVendor::Amd => score += 50,
                GpuVendor::Intel => score += 30,
                _ => {}
            }
        }
    }

    score
}

fn select_gpu_with_preference<'a>(
    candidates: &'a [GpuCandidate],
    preference: &GpuPreference,
) -> Result<&'a GpuCandidate> {
    if candidates.is_empty() {
        return Err(anyhow::anyhow!("No GPU candidates available"));
    }

    let selected = match preference {
        GpuPreference::Auto => select_auto(candidates),
        GpuPreference::HighPerformance => select_high_performance(candidates),
        GpuPreference::PowerEfficient => select_power_efficient(candidates),
        GpuPreference::Specific(name) => select_specific(candidates, name),
    }?;

    Ok(selected)
}

fn select_auto(candidates: &[GpuCandidate]) -> Result<&GpuCandidate> {
    // Check if we have any desktop dedicated GPUs (indicates desktop system)
    let has_desktop_dedicated = candidates.iter().any(|c| {
        matches!(c.gpu_type, GpuType::Dedicated { .. })
            && matches!(c.mobility, GpuMobility::Desktop)
    });

    if has_desktop_dedicated {
        // Desktop system: prefer dedicated desktop GPUs for performance
        candidates
            .iter()
            .filter(|c| matches!(c.gpu_type, GpuType::Dedicated { .. }))
            .filter(|c| matches!(c.mobility, GpuMobility::Desktop))
            .max_by_key(|c| c.score)
            .or_else(|| candidates.iter().max_by_key(|c| c.score))
    } else {
        // Laptop system: prefer power efficiency (integrated first, then mobile)
        candidates
            .iter()
            .min_by_key(|c| match (&c.gpu_type, &c.mobility) {
                (GpuType::Integrated { .. }, _) => 0, // Integrated first
                (GpuType::Dedicated { .. }, GpuMobility::Mobile) => 1, // Mobile dedicated second
                (GpuType::Dedicated { .. }, GpuMobility::Desktop) => 2, // Fallback to desktop dedicated
                _ => 3,                                                 // Unknown last
            })
    }
    .ok_or_else(|| anyhow::anyhow!("No suitable GPU found"))
}

fn select_high_performance(candidates: &[GpuCandidate]) -> Result<&GpuCandidate> {
    // Force dedicated GPUs only
    candidates
        .iter()
        .filter(|c| matches!(c.gpu_type, GpuType::Dedicated { .. }))
        .max_by_key(|c| c.score)
        .or_else(|| candidates.first()) // Fallback to any GPU
        .ok_or_else(|| anyhow::anyhow!("No dedicated GPU found"))
}

fn select_power_efficient(candidates: &[GpuCandidate]) -> Result<&GpuCandidate> {
    // Prefer integrated, then mobile, then desktop
    candidates
        .iter()
        .min_by_key(|c| match c.mobility {
            GpuMobility::Integrated => 0,
            GpuMobility::Mobile => 1,
            GpuMobility::Desktop => 2,
            GpuMobility::Unknown => 3,
        })
        .ok_or_else(|| anyhow::anyhow!("No power-efficient GPU found"))
}

fn select_specific<'a>(
    candidates: &'a [GpuCandidate],
    target_name: &str,
) -> Result<&'a GpuCandidate> {
    let target_lower = target_name.to_lowercase();

    candidates
        .iter()
        .find(|c| c.info.device_name.to_lowercase().contains(&target_lower))
        .or_else(|| candidates.first()) // Fallback to best available
        .ok_or_else(|| anyhow::anyhow!("Specific GPU '{}' not found", target_name))
}

fn log_available_gpus(candidates: &[GpuCandidate]) {
    log::info!("Available GPUs:");
    for (i, candidate) in candidates.iter().enumerate() {
        let dedicated_mb = candidate.info.dedicated_memory / (1024 * 1024);
        let shared_mb = candidate.info.shared_memory / (1024 * 1024);

        let gpu_type_str = match &candidate.gpu_type {
            GpuType::Dedicated {
                vendor,
                memory_mb: _,
            } => {
                format!(
                    "Dedicated {:?} ({}MB VRAM, {}MB shared)",
                    vendor, dedicated_mb, shared_mb
                )
            }
            GpuType::Integrated { vendor } => {
                format!("Integrated {:?} ({}MB shared)", vendor, shared_mb)
            }
        };
        log::info!(
            "  {}: {} - {} {:?} (Score: {}) [VID:{:#06X} DID:{:#06X}]",
            i,
            candidate.info.device_name,
            gpu_type_str,
            candidate.mobility,
            candidate.score,
            candidate.info.vendor_id,
            candidate.info.device_id
        );
    }
}
