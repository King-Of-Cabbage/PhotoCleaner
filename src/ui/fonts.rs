use std::fs;
use std::path::Path;

use egui::{Context, FontData, FontDefinitions, FontFamily};

const WINDOWS_CJK_FONTS: &[&str] = &[
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\msyh.ttf",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\simsun.ttc",
];

pub fn install_chinese_fonts(ctx: &Context) {
    let Some((name, data)) = load_first_available_font() else {
        return;
    };

    let mut fonts = FontDefinitions::default();
    fonts
        .font_data
        .insert(name.clone(), FontData::from_owned(data));

    for family in [FontFamily::Proportional, FontFamily::Monospace] {
        fonts
            .families
            .entry(family)
            .or_default()
            .insert(0, name.clone());
    }

    ctx.set_fonts(fonts);
}

fn load_first_available_font() -> Option<(String, Vec<u8>)> {
    WINDOWS_CJK_FONTS.iter().find_map(|path| {
        fs::read(path)
            .ok()
            .map(|data| (font_name_from_path(path), data))
    })
}

fn font_name_from_path(path: &str) -> String {
    Path::new(path)
        .file_stem()
        .and_then(|name| name.to_str())
        .unwrap_or("windows_cjk")
        .to_string()
}
