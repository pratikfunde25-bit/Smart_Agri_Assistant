import json
from pathlib import Path


NOTEBOOK = Path(r"c:\Users\Pratik\Downloads\cop\crop-recommendation-system.ipynb")


def md_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def mojibake(text: str) -> str:
    return text.encode("utf-8").decode("latin1")


REPLACEMENTS = {
    mojibake("🐍 "): "",
    mojibake("🤖 "): "",
    mojibake("🌿 "): "",
    mojibake("📊 "): "",
    mojibake("📌 "): "",
    mojibake("🧬 "): "",
    mojibake("⚗️ "): "",
    mojibake("⚙️ "): "",
    mojibake("🛠️ "): "",
    mojibake("📐 "): "",
    mojibake("🔁 "): "",
    mojibake("🍩 "): "",
    mojibake("💧 "): "",
    mojibake("🌡️ "): "",
    mojibake("🌧️ "): "",
    mojibake("🎯 "): "",
    mojibake("📦 "): "",
    mojibake("💼 "): "",
    mojibake("✍️ "): "",
    mojibake("🎉"): "",
    mojibake("⚖️ "): "",
    mojibake("🌟 "): "",
    mojibake("🏆 "): "",
    mojibake("🔲 "): "",
    mojibake("📋 "): "",
    mojibake("📈 "): "",
    mojibake("📂 "): "",
    mojibake("📚 "): "",
    mojibake("💾 "): "",
    mojibake("🏷️ "): "",
    mojibake("✍️"): "Author",
    mojibake("⚗️"): "",
    mojibake("🌡️"): "",
    mojibake("🌧️"): "",
    mojibake("⚙️"): "",
    mojibake("🛠️"): "",
    mojibake("🏷️"): "",
    mojibake("→"): "->",
    mojibake("—"): "-",
    mojibake("·"): "·",
    mojibake("©"): "©",
    "Â°C": "°C",
}


def clean_text(text: str) -> str:
    for old, new in REPLACEMENTS.items():
        text = text.replace(old, new)
    return text


def add_advanced_eda_cells(nb: dict) -> None:
    if any("Advanced EDA Deep Dive" in "".join(c.get("source", [])) for c in nb["cells"]):
        return

    anchor = None
    for i, cell in enumerate(nb["cells"]):
        src = "".join(cell.get("source", []))
        if "4. Feature Engineering" in src:
            anchor = i
            break

    if anchor is None:
        anchor = 11

    new_cells = [
        md_cell(
            """<div style="background: linear-gradient(90deg, #0f4c5c 0%, #1b6b75 100%); border-radius: 10px; padding: 14px 24px; margin-bottom: 20px; box-shadow: 0 4px 10px rgba(0,0,0,0.15);">
    <h2 style="margin: 0; color: white; font-size: 22px; font-weight: 700;">
        3.1 Advanced EDA Deep Dive
    </h2>
</div>

These additional visuals are designed to show a deeper analytical understanding of the crop recommendation dataset and strengthen the project presentation."""
        ),
        code_cell(
            """# Advanced EDA 1: Project KPI summary cards
kpi_html = f'''
<div style="display:grid; grid-template-columns: repeat(4, 1fr); gap:14px; margin: 12px 0 18px;">
  <div style="background:#e8f5e9; border-radius:14px; padding:18px; border:1px solid #c8e6c9;">
    <div style="font-size:12px; color:#2e7d32; font-weight:700;">TOTAL RECORDS</div>
    <div style="font-size:30px; font-weight:800; color:#1b5e20;">{len(df):,}</div>
  </div>
  <div style="background:#e3f2fd; border-radius:14px; padding:18px; border:1px solid #bbdefb;">
    <div style="font-size:12px; color:#1565c0; font-weight:700;">CROP CLASSES</div>
    <div style="font-size:30px; font-weight:800; color:#0d47a1;">{df['label'].nunique()}</div>
  </div>
  <div style="background:#fff3e0; border-radius:14px; padding:18px; border:1px solid #ffe0b2;">
    <div style="font-size:12px; color:#ef6c00; font-weight:700;">AVG RAINFALL</div>
    <div style="font-size:30px; font-weight:800; color:#e65100;">{df['rainfall'].mean():.1f}</div>
  </div>
  <div style="background:#f3e5f5; border-radius:14px; padding:18px; border:1px solid #e1bee7;">
    <div style="font-size:12px; color:#7b1fa2; font-weight:700;">AVG TEMPERATURE</div>
    <div style="font-size:30px; font-weight:800; color:#4a148c;">{df['temperature'].mean():.1f}°C</div>
  </div>
</div>
'''
display(HTML(kpi_html))"""
        ),
        code_cell(
            """# Advanced EDA 2: Soil health score by crop
df['soil_health_score'] = (
    0.30 * df['N'] +
    0.25 * df['P'] +
    0.25 * df['K'] +
    0.20 * (14 - abs(df['ph'] - 7) * 2)
)

soil_rank = (
    df.groupby('label')['soil_health_score']
      .mean()
      .sort_values(ascending=False)
      .head(12)
      .sort_values()
)

fig, ax = plt.subplots(figsize=(12, 7), facecolor=BG)
ax.set_facecolor(BG)
bars = ax.barh(soil_rank.index, soil_rank.values, color=sns.color_palette('Greens', 12))
ax.set_title('Soil Health Score - Top Crops', fontsize=18, fontweight='bold', color=GREEN, pad=14)
ax.set_xlabel('Average Soil Health Score', fontsize=12)
ax.grid(axis='x', color='#dfe6e9', linewidth=1, alpha=0.8)
for bar in bars:
    ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2, f'{bar.get_width():.1f}',
            va='center', fontsize=10, color='#37474f', fontweight='bold')
plt.tight_layout()
plt.show()"""
        ),
        code_cell(
            """# Advanced EDA 3: Crop-wise climate zone heatmap
df['temp_band'] = pd.cut(df['temperature'], bins=[0, 20, 25, 30, 40], labels=['Cool', 'Mild', 'Warm', 'Hot'])
df['humidity_band'] = pd.cut(df['humidity'], bins=[0, 40, 60, 80, 100], labels=['Low', 'Moderate', 'Humid', 'Very Humid'])

zone_table = pd.crosstab(df['label'], [df['temp_band'], df['humidity_band']])
top_crops = df['label'].value_counts().index[:10]
zone_table = zone_table.loc[top_crops]

fig, ax = plt.subplots(figsize=(14, 8), facecolor=BG)
ax.set_facecolor(BG)
sns.heatmap(zone_table, cmap='YlGnBu', linewidths=0.5, linecolor='white', ax=ax)
ax.set_title('Climate Zone Distribution Across Major Crops', fontsize=18, fontweight='bold', color=GREEN, pad=16)
ax.set_xlabel('Temperature and Humidity Bands', fontsize=12)
ax.set_ylabel('Crop', fontsize=12)
plt.tight_layout()
plt.show()"""
        ),
        code_cell(
            """# Advanced EDA 4: PCA-based crop separation
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

pca_features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
X_scaled = StandardScaler().fit_transform(df[pca_features])
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
pca_df['label'] = df['label'].values
plot_crops = df['label'].value_counts().head(8).index

fig, ax = plt.subplots(figsize=(12, 8), facecolor=BG)
ax.set_facecolor(BG)
sns.scatterplot(
    data=pca_df[pca_df['label'].isin(plot_crops)],
    x='PC1', y='PC2', hue='label', palette='tab10', s=70, alpha=0.85, ax=ax
)
ax.set_title('PCA Projection of Crop Profiles', fontsize=18, fontweight='bold', color=GREEN, pad=16)
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)')
ax.legend(title='Crop', bbox_to_anchor=(1.02, 1), loc='upper left')
ax.grid(color='#e8f5e9', linewidth=1)
plt.tight_layout()
plt.show()"""
        ),
        code_cell(
            """# Advanced EDA 5: Rainfall vs pH crop landscape
focus_crops = ['rice', 'maize', 'cotton', 'jute', 'apple', 'grapes']
plot_df = df[df['label'].isin(focus_crops)]

fig, ax = plt.subplots(figsize=(12, 7), facecolor=BG)
ax.set_facecolor(BG)
sns.scatterplot(
    data=plot_df, x='rainfall', y='ph', size='humidity', hue='label',
    palette='Set2', sizes=(40, 220), alpha=0.75, ax=ax
)
ax.set_title('Rainfall vs pH Landscape for Key Crops', fontsize=18, fontweight='bold', color=GREEN, pad=16)
ax.set_xlabel('Rainfall (mm)')
ax.set_ylabel('Soil pH')
ax.grid(color='#eceff1', linewidth=1)
plt.tight_layout()
plt.show()"""
        ),
        code_cell(
            """# Advanced EDA 6: Representative crop comparison radar chart
representative_crops = ['rice', 'maize', 'cotton', 'apple', 'grapes']
radar_df = df[df['label'].isin(representative_crops)].groupby('label')[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']].mean()
scaled_radar = (radar_df - radar_df.min()) / (radar_df.max() - radar_df.min())

categories = scaled_radar.columns.tolist()
angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
angles += angles[:1]

fig = plt.figure(figsize=(10, 10), facecolor=BG)
ax = plt.subplot(111, polar=True)
ax.set_facecolor('#fcfdfc')

for crop in scaled_radar.index:
    values = scaled_radar.loc[crop].tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=2, label=crop)
    ax.fill(angles, values, alpha=0.08)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=11)
ax.set_yticklabels([])
ax.set_title('Representative Crop Signature Radar Chart', fontsize=18, fontweight='bold', color=GREEN, pad=22)
ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1))
plt.tight_layout()
plt.show()"""
        ),
    ]

    nb["cells"][anchor:anchor] = new_cells


def main() -> None:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))

    for cell in nb["cells"]:
        if "source" in cell:
            src = "".join(cell["source"])
            cell["source"] = clean_text(src).splitlines(keepends=True)

    add_advanced_eda_cells(nb)

    NOTEBOOK.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
