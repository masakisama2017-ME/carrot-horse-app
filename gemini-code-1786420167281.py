import streamlit as st
import json
import os
import pandas as pd

# Gemini API ライブラリのインポートチェック
try:
    from google import genai
    from google.genai import types
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

# ページ基本設定
st.set_page_config(page_title="キャロットクラブ AI募集馬診断 2026", layout="wide", page_icon="🐴")

# -----------------------------------------------------------------------------
# データ読み込み関数
# -----------------------------------------------------------------------------
@st.cache_data
def load_and_process_data():
    if not os.path.exists("horses.json"):
        return {}, pd.DataFrame()
        
    with open("horses.json", "r", encoding="utf-8") as f:
        raw_db = json.load(f)

    processed_list = []
    
    for name, data in raw_db.items():
        birth_year = int(data.get("birth_date", "2025-01-01").split("-")[0])
        mother_birth_year = data.get("mother_birth_year")
        mother_age = (birth_year - mother_birth_year) if mother_birth_year else None
        
        siblings = data.get("siblings", [])
        consecutive_count = 1
        for i in range(1, 10):
            if any(s.get("birth_year") == birth_year - i for s in siblings):
                consecutive_count += 1
            else:
                break
        
        total_price = data.get("total_price", 0)
        total_price_num = total_price if isinstance(total_price, (int, float)) else 0
        
        stud_fee = data.get("stud_fee", 0)
        stud_fee_num = stud_fee if isinstance(stud_fee, (int, float)) else 0
        
        price_ratio = round(total_price_num / stud_fee_num, 1) if stud_fee_num > 0 and total_price_num > 0 else None

        meas = data.get("measurements", {})
        try:
            cannon_cm = float(str(meas.get("cannon", "")).replace("cm", "").strip())
        except (ValueError, TypeError):
            cannon_cm = None
            
        try:
            weight_kg = float(str(meas.get("weight", "")).replace("kg", "").strip())
        except (ValueError, TypeError):
            weight_kg = None

        processed_list.append({
            "募集馬名": name,
            "父": data.get("father", "不明"),
            "母": data.get("mother", "不明"),
            "母父": data.get("mother_father", "不明"),
            "性別": data.get("sex", "不明"),
            "厩舎": data.get("trainer", "未定"),
            "募集総額(万円)": total_price_num,
            "一口価格(万円)": data.get("unit_price", 0.0),
            "種付け料(万円)": stud_fee_num if stud_fee_num > 0 else "未定",
            "価格/種付け料倍率": price_ratio,
            "母出産時年齢": mother_age,
            "連産数": consecutive_count,
            "第何仔": len(siblings) + 1,
            "管囲(cm)": cannon_cm,
            "馬体重(kg)": weight_kg,
            "raw_data": data
        })

    df = pd.DataFrame(processed_list)
    return raw_db, df

horse_db, df_horses = load_and_process_data()

# -----------------------------------------------------------------------------
# サイドバー: APIキー設定とアプリ共通設定
# -----------------------------------------------------------------------------
st.sidebar.title("⚙️ 設定 & APIキー")

# APIキーの取得（環境変数 または サイドバー入力）
env_api_key = os.environ.get("GEMINI_API_KEY", "")
api_key_input = st.sidebar.text_input(
    "Google AI Studio API Key",
    value=env_api_key,
    type="password",
    help="Google AI Studioで作成したGemini APIキーを入力してください。"
)

api_key = api_key_input if api_key_input else env_api_key

if api_key:
    st.sidebar.success("🔑 APIキー設定済み")
else:
    st.sidebar.warning("⚠️ APIキー未設定\nAI診断機能を使用するにはAPIキーを入力してください。")

st.sidebar.markdown("---")
st.sidebar.info("💡 **Google AI Studio**で無料発行したAPIキーで、Gemini 2.5 Flashモデルが馬体・血統を詳細分析します。")

# -----------------------------------------------------------------------------
# メイン画面
# -----------------------------------------------------------------------------
st.title("🐴 キャロットクラブ 募集馬データベース 2026")

tab1, tab2, tab3 = st.tabs(["🔍 条件検索・比較", "📌 個別詳細 & AI診断", "🤖 全馬一括AIスクリーニング"])

# ==========================================
# TAB 1: 条件検索一覧
# ==========================================
with tab1:
    st.subheader("🎯 募集馬フィルター検索")

    if df_horses.empty:
        st.warning("データが見つかりません。`horses.json` を確認してください。")
    else:
        with st.expander("🛠️ 絞り込み条件パネル", expanded=True):
            f_col1, f_col2, f_col3, f_col4 = st.columns(4)

            with f_col1:
                sex_options = df_horses["性別"].unique().tolist()
                selected_sex = st.multiselect("性別", options=sex_options, default=sex_options)

                sire_options = sorted(df_horses["父"].unique().tolist())
                selected_sires = st.multiselect("父馬", options=sire_options, default=sire_options)

            with f_col2:
                # 募集総額フィルター（1000万円未満の場合は1000に固定して安全保護）
                raw_max_price = df_horses["募集総額(万円)"].max() if not df_horses.empty else 10000
                max_price = max(int(raw_max_price) if pd.notna(raw_max_price) and raw_max_price > 0 else 10000, 1000)
                price_range = st.slider("募集総額（万円）", 0, max_price, (0, max_price), step=500)

                # 母の出産時年齢フィルター
                valid_ages = df_horses["母出産時年齢"].dropna()
                min_a = int(valid_ages.min()) if not valid_ages.empty else 4
                max_a = max(int(valid_ages.max()) if not valid_ages.empty else 20, min_a + 1)
                age_range = st.slider("母の出産時年齢", min_a, max_a, (min_a, max_a))

            with f_col3:
                # 連産数フィルター（1未満にならないよう保護）
                raw_max_consec = df_horses["連産数"].max() if not df_horses.empty else 1
                max_consec = max(int(raw_max_consec) if pd.notna(raw_max_consec) and raw_max_consec > 0 else 1, 1)
                consec_limit = st.slider("連産数の上限（〜連産目）", 1, max_consec, max_consec)

                min_cannon = st.number_input("最小管囲（cm以上）", min_value=0.0, max_value=25.0, value=0.0, step=0.1)
                include_no_meas = st.checkbox("測尺未発表も含める", value=True)

            with f_col4:
                sort_col = st.selectbox(
                    "並び替え項目",
                    options=["募集総額(万円)", "母出産時年齢", "連産数", "管囲(cm)", "価格/種付け料倍率"],
                    index=0
                )
                sort_order = st.radio("順序", options=["昇順（順）", "降順（逆順）"], horizontal=True)

        filtered_df = df_horses[
            (df_horses["性別"].isin(selected_sex)) &
            (df_horses["父"].isin(selected_sires)) &
            (df_horses["募集総額(万円)"] >= price_range[0]) &
            (df_horses["募集総額(万円)"] <= price_range[1]) &
            (df_horses["母出産時年齢"].fillna(0) >= age_range[0]) &
            (df_horses["母出産時年齢"].fillna(99) <= age_range[1]) &
            (df_horses["連産数"] <= consec_limit)
        ]

        if min_cannon > 0:
            if include_no_meas:
                filtered_df = filtered_df[(filtered_df["管囲(cm)"] >= min_cannon) | (filtered_df["管囲(cm)"].isna())]
            else:
                filtered_df = filtered_df[filtered_df["管囲(cm)"] >= min_cannon]

        ascending = True if "昇順" in sort_order else False
        filtered_df = filtered_df.sort_values(by=sort_col, ascending=ascending, na_position="last")

        st.write(f"該当件数: **{len(filtered_df)}** 件 / 全 {len(df_horses)} 件")

        display_cols = [
            "募集馬名", "父", "母", "性別", "厩舎",
            "募集総額(万円)", "一口価格(万円)", "種付け料(万円)", "価格/種付け料倍率",
            "母出産時年齢", "連産数", "管囲(cm)"
        ]

        st.dataframe(filtered_df[display_cols], use_container_width=True, hide_index=True, height=400)

# ==========================================
# TAB 2: 個別詳細 & AI診断機能
# ==========================================
with tab2:
    st.subheader("📌 募集馬 個別詳細 ＆ 🤖 Gemini AI分析")
    
    horse_names = list(horse_db.keys())
    if horse_names:
        selected_horse = st.selectbox("分析・閲覧する募集馬を選択:", options=horse_names)
        
        if selected_horse in horse_db:
            data = horse_db[selected_horse]
            
            birth_year = int(data.get("birth_date", "2025-01-01").split("-")[0])
            mother_age = birth_year - data["mother_birth_year"] if data.get("mother_birth_year") else "不明"
            
            siblings = data.get("siblings", [])
            consecutive_count = 1
            for i in range(1, 10):
                if any(s.get("birth_year") == birth_year - i for s in siblings):
                    consecutive_count += 1
                else:
                    break
            child_order = len(siblings) + 1

            # メトリクス表示
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("母の出産時年齢", f"{mother_age} 歳")
            
            consec_label = f"{child_order}番目の仔 ({consecutive_count}連産目)" if consecutive_count > 1 else f"{child_order}番目の仔 (空胎明け)"
            m2.metric("連産状況", consec_label)
            
            stud_fee_val = data.get('stud_fee', '未定')
            stud_fee_text = f"{stud_fee_val} 万円" if isinstance(stud_fee_val, (int, float)) else str(stud_fee_val)
            m3.metric("父の種付け料", stud_fee_text)
            
            if isinstance(data.get("total_price"), (int, float)) and isinstance(data.get("stud_fee"), (int, float)) and data["stud_fee"] > 0:
                ratio = round(data["total_price"] / data["stud_fee"], 1)
                m4.metric("価格/種付け料 倍率", f"{ratio} 倍")
            else:
                m4.metric("価格/種付け料 倍率", "未定")

            st.divider()

            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"### 📋 基本情報 ({data['father']} × {data['mother']})")
                st.markdown(f"- **性別・毛色:** {data['sex']} / {data.get('color', '不明')}")
                st.markdown(f"- **生年月日:** {data.get('birth_date')}")
                st.markdown(f"- **予定厩舎:** {data.get('trainer')}")
                tot_p = f"{data['total_price']} 万円" if isinstance(data.get('total_price'), (int, float)) else data.get('total_price')
                st.markdown(f"- **募集価格:** 総額 {tot_p}")
                
                st.markdown("### 📏 測尺データ")
                m = data.get('measurements', {})
                st.text(f"馬体重: {m.get('weight', '---')} kg | 体高: {m.get('height', '---')} cm\n胸囲: {m.get('chest', '---')} cm   | 管囲: {m.get('cannon', '---')} cm")

            with col2:
                st.markdown("### 🧬 血統・母系成績")
                st.markdown(f"- **母父:** {data.get('mother_father')}")
                st.markdown(f"- **母競走成績:** {data.get('mother_record')}")
                
                st.markdown("**兄弟馬の状況:**")
                if siblings:
                    for sib in siblings:
                        st.markdown(f"  - {sib['name']}（{sib['birth_year']}年産 / 父: {sib['father']}）: {sib['record']}")
                else:
                    st.markdown("  - 初仔")

            st.divider()

            # -------------------------------------------------------------
            # AI 診断セクション
            # -------------------------------------------------------------
            st.subheader("🤖 Gemini AI 血統 & 馬体詳細診断")
            
            if not api_key:
                st.info("👈 左側のサイドバーに Google AI Studio の API キーを入力すると、AI診断が実行できます。")
            else:
                ai_btn = st.button("✨ この馬のAI診断を実行する", type="primary")
                
                if ai_btn:
                    with st.spinner("Gemini AIが血統・測尺・母系データを多角的分析中..."):
                        prompt_content = f"""
あなたは一口馬主（特にキャロットクラブ）のプロ馬券師・競走馬評価エキスパートです。
以下の募集馬データに基づき、プロの観点から詳細な出資診断を行ってください。

【対象馬】: {selected_horse}
- 父: {data.get('father')} (種付け料: {data.get('stud_fee')}万円)
- 母: {data.get('mother')} (母の生年: {data.get('mother_birth_year')}年 / 出産時年齢: {mother_age}歳)
- 母父: {data.get('mother_father')}
- 性別・毛色: {data.get('sex')} / {data.get('color')}
- 生年月日: {data.get('birth_date')}
- 予定厩舎: {data.get('trainer')}
- 募集価格: 総額 {data.get('total_price')}万円 (一口 {data.get('unit_price')}万円)
- 測尺: 馬体重 {m.get('weight')}kg, 体高 {m.get('height')}cm, 胸囲 {m.get('chest')}cm, 管囲 {m.get('cannon')}cm
- 母の競走成績: {data.get('mother_record')}
- 兄弟の成績: {json.dumps(siblings, ensure_ascii=False)}
- 近親の活躍馬: {json.dumps(data.get('notable_relatives', []), ensure_ascii=False)}
- 連産状況: 第{child_order}仔 ({consecutive_count}連産目)

以下の項目について、わかりやすく構造化された見出し・マークダウンで回答してください：

1. **適性予測 (芝/ダート & 距離適性)**
   - 父・母父の血統構成と母系から推測されるベストな馬場（芝・ダート）および距離適性範囲。

2. **予想デビュー時馬体重 & 馬体評価**
   - 生年月と現在の馬体重、管囲から、3歳デビュー時の予想馬体重範囲と、脚元のリスク（管囲の太さ評価など）を分析。

3. **勝ち上がり期待度 (S/A/B/C/D)**
   - 評価ランクと、その理由（母の出産年齢、連産疲労の有無、血統のニックス、兄弟の勝ち上がり率など）。

4. **コストパフォーマンス評価**
   - 種付け料に対して募集総額の上乗せが妥当か、回収期待度。

5. **総合アドバイス (一口馬主としての出資判断)**
   - どのような出資スタイルの人に向いているか。
"""
                        try:
                            if HAS_GENAI:
                                client = genai.Client(api_key=api_key)
                                response = client.models.generate_content(
                                    model='gemini-2.5-flash',
                                    contents=prompt_content,
                                )
                                ai_result = response.text
                            else:
                                import google.generativeai as legacy_genai
                                legacy_genai.configure(api_key=api_key)
                                model = legacy_genai.GenerativeModel('gemini-1.5-flash')
                                response = model.generate_content(prompt_content)
                                ai_result = response.text
                                
                            st.markdown("### 📊 AI診断結果")
                            st.info(ai_result)
                            
                        except Exception as e:
                            st.error(f"AI診断の実行中にエラーが発生しました: {e}")

# ==========================================
# TAB 3: 全馬一括AIスクリーニング
# ==========================================
with tab3:
    st.subheader("🤖 全馬一括 AIお買い得スクリーニング")
    st.markdown("登録されている全馬のデータをGemini AIに渡し、**「回収率が期待できるおすすめTOP3」** を選出させます。")
    
    if not api_key:
        st.info("👈 左側のサイドバーに Google AI Studio の API キーを入力してください。")
    else:
        if st.button("🚀 全馬一括スクリーニングを実行", type="primary"):
            with st.spinner("Gemini AIが全募集馬のデータ（価格・血統・母年齢・管囲）を一括比較分析中..."):
                all_horses_summary = []
                for h_name, h_data in horse_db.items():
                    all_horses_summary.append({
                        "馬名": h_name,
                        "父": h_data.get("father"),
                        "母父": h_data.get("mother_father"),
                        "募集価格": h_data.get("total_price"),
                        "種付け料": h_data.get("stud_fee"),
                        "母生年": h_data.get("mother_birth_year"),
                        "管囲": h_data.get("measurements", {}).get("cannon"),
                        "兄弟成績": [s.get("record") for s in h_data.get("siblings", [])]
                    })
                
                batch_prompt = f"""
以下は2026年度キャロットクラブ募集馬のデータ一覧です。
プロの一口馬主分析官として、全馬を比較評価し、**最も回収率・お買い得感・勝ち上がり期待度が高い「おすすめ馬TOP3」** を選出してください。

【募集馬データリスト】:
{json.dumps(all_horses_summary, ensure_ascii=False, indent=2)}

回答形式：
1. **おすすめ第1位：馬名**
   - 選定理由（血統、コスパ、母年齢、管囲などの強み）
2. **おすすめ第2位：馬名**
   - 選定理由
3. **おすすめ第3位：馬名**
   - 選定理由
4. **全体総評・今年度の募集傾向アドバイス**
"""
                try:
                    if HAS_GENAI:
                        client = genai.Client(api_key=api_key)
                        response = client.models.generate_content(
                            model='gemini-2.5-flash',
                            contents=batch_prompt,
                        )
                        batch_result = response.text
                    else:
                        import google.generativeai as legacy_genai
                        legacy_genai.configure(api_key=api_key)
                        model = legacy_genai.GenerativeModel('gemini-1.5-flash')
                        response = model.generate_content(batch_prompt)
                        batch_result = response.text

                    st.success("分析が完了しました！")
                    st.markdown(batch_result)
                except Exception as e:
                    st.error(f"エラーが発生しました: {e}")
