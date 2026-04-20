use std::{path::Path, sync::Arc};

use fuzzy::StringMatchCandidate;
use futures::channel::oneshot;
use gpui::{App, Context, DismissEvent, Entity, EventEmitter, Focusable, Subscription, Task, Window};
use picker::{Picker, PickerDelegate};
use ui::{
    HighlightedLabel, InteractiveElement, LabelCommon, LabelSize, ListItem, ListItemSpacing,
    ParentElement, Render, Styled, Toggleable, div, prelude::*, rems, v_flex,
};
use workspace::ModalView;

pub struct SettingsMigrationModal {
    picker: Entity<Picker<SettingsMigrationDelegate>>,
    _subscriptions: Vec<Subscription>,
}

impl SettingsMigrationModal {
    pub fn new(
        folder_paths: Vec<Arc<Path>>,
        sender: oneshot::Sender<Arc<Path>>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Self {
        let delegate = SettingsMigrationDelegate::new(folder_paths, sender);
        let picker = cx.new(|cx| Picker::uniform_list(delegate, window, cx).modal(false));

        let subscriptions = vec![
            cx.subscribe_in(&picker, window, |_, _, _: &DismissEvent, _, cx| {
                cx.emit(DismissEvent);
            }),
            cx.subscribe_in(
                &picker,
                window,
                |_, _, _: &SettingsMigrationConfirmed, _, cx| {
                    cx.emit(DismissEvent);
                },
            ),
        ];

        Self {
            picker,
            _subscriptions: subscriptions,
        }
    }
}

impl ModalView for SettingsMigrationModal {}

impl Focusable for SettingsMigrationModal {
    fn focus_handle(&self, cx: &App) -> gpui::FocusHandle {
        self.picker.focus_handle(cx)
    }
}

impl EventEmitter<DismissEvent> for SettingsMigrationModal {}

impl Render for SettingsMigrationModal {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        div()
            .on_mouse_down_out(cx.listener(|_, _, _, cx| cx.emit(DismissEvent)))
            .elevation_3(cx)
            .w(rems(34.))
            .child(
                v_flex()
                    .px_2()
                    .pt_1()
                    .child(
                        ui::Label::new("Copy project settings (.zed) to:")
                            .size(LabelSize::Small)
                            .color(ui::Color::Muted),
                    )
                    .child(self.picker.clone()),
            )
    }
}

#[derive(Clone, Debug)]
struct SettingsMigrationConfirmed;

impl EventEmitter<SettingsMigrationConfirmed> for Picker<SettingsMigrationDelegate> {}

pub struct SettingsMigrationDelegate {
    folder_paths: Vec<Arc<Path>>,
    folder_names: Vec<String>,
    matches: Vec<fuzzy::StringMatch>,
    selected_index: usize,
    sender: Option<oneshot::Sender<Arc<Path>>>,
}

impl SettingsMigrationDelegate {
    fn new(folder_paths: Vec<Arc<Path>>, sender: oneshot::Sender<Arc<Path>>) -> Self {
        let folder_names: Vec<String> = folder_paths
            .iter()
            .map(|path| {
                path.file_name()
                    .map(|name| name.to_string_lossy().into_owned())
                    .unwrap_or_else(|| path.to_string_lossy().into_owned())
            })
            .collect();

        Self {
            folder_paths,
            folder_names,
            matches: Vec::new(),
            selected_index: 0,
            sender: Some(sender),
        }
    }
}

impl PickerDelegate for SettingsMigrationDelegate {
    type ListItem = ListItem;

    fn match_count(&self) -> usize {
        self.matches.len()
    }

    fn selected_index(&self) -> usize {
        self.selected_index
    }

    fn set_selected_index(
        &mut self,
        ix: usize,
        _window: &mut Window,
        cx: &mut Context<Picker<Self>>,
    ) {
        self.selected_index = ix;
        cx.notify();
    }

    fn placeholder_text(&self, _window: &mut Window, _cx: &mut App) -> Arc<str> {
        Arc::from("Select a folder...")
    }

    fn update_matches(
        &mut self,
        query: String,
        _window: &mut Window,
        cx: &mut Context<Picker<Self>>,
    ) -> Task<()> {
        let candidates: Vec<StringMatchCandidate> = self
            .folder_names
            .iter()
            .enumerate()
            .map(|(id, name)| StringMatchCandidate::new(id, name))
            .collect();

        let query = query.trim_start().to_string();
        if query.is_empty() {
            self.matches = candidates
                .iter()
                .map(|candidate| fuzzy::StringMatch {
                    candidate_id: candidate.id,
                    score: 0.0,
                    positions: Vec::new(),
                    string: candidate.string.clone(),
                })
                .collect();
        } else {
            let smart_case = query.chars().any(|c| c.is_uppercase());
            self.matches = smol::block_on(fuzzy::match_strings(
                &candidates,
                &query,
                smart_case,
                true,
                100,
                &Default::default(),
                cx.background_executor().clone(),
            ));
        }
        self.selected_index = 0;
        Task::ready(())
    }

    fn confirm(&mut self, _secondary: bool, _window: &mut Window, cx: &mut Context<Picker<Self>>) {
        if let Some(matched) = self.matches.get(self.selected_index) {
            if let Some(sender) = self.sender.take() {
                let path = self.folder_paths[matched.candidate_id].clone();
                let _ = sender.send(path);
            }
            cx.emit(SettingsMigrationConfirmed);
        }
    }

    fn dismissed(&mut self, _window: &mut Window, cx: &mut Context<Picker<Self>>) {
        self.sender.take();
        cx.emit(DismissEvent);
    }

    fn render_match(
        &self,
        ix: usize,
        selected: bool,
        _window: &mut Window,
        _cx: &mut Context<Picker<Self>>,
    ) -> Option<Self::ListItem> {
        let matched = self.matches.get(ix)?;
        let full_path = &self.folder_paths[matched.candidate_id];
        Some(
            ListItem::new(ix)
                .toggle_state(selected)
                .inset(true)
                .spacing(ListItemSpacing::Sparse)
                .child(
                    v_flex()
                        .child(HighlightedLabel::new(
                            matched.string.clone(),
                            matched.positions.clone(),
                        ))
                        .child(
                            ui::Label::new(full_path.to_string_lossy().to_string())
                                .size(LabelSize::XSmall)
                                .color(ui::Color::Muted),
                        ),
                ),
        )
    }
}
