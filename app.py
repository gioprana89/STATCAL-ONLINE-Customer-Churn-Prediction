"""STATCAL ONLINE — Machine Learning Lab | Praktikum 4 Pertanyaan.

Jalankan: python -m streamlit run app.py
Dataset default: dataset_pelanggan.csv (sintetis).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from ml_core import (
    DEFAULT_TARGET, GUIDE, ID_COLUMNS,
    create_excel_report, predict_customer_profile, run_logistic_analysis,
)

APP_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = APP_DIR / "dataset_pelanggan.csv"


def fmt_pct(v):
    return f"{v:.1%}"


def confusion_plot(matrix, negative, positive):
    fig, ax = plt.subplots(figsize=(5.9, 3.9))
    ax.imshow(matrix, cmap="Blues", vmin=0)
    for row in range(2):
        for col in range(2):
            value = matrix[row, col]
            tag = [["TN", "FP"], ["FN", "TP"]][row][col]
            ax.text(col, row, f"{tag}\n{value}", ha="center", va="center",
                    color="#0B2A43", fontsize=16, weight="bold")
    ax.set_xticks([0, 1], [negative, positive])
    ax.set_yticks([0, 1], [negative, positive])
    ax.set_xlabel("Prediksi model")
    ax.set_ylabel("Kondisi sebenarnya")
    ax.set_title("Confusion Matrix — Hanya Data Testing")
    fig.tight_layout()
    return fig


def start_app():
    st.set_page_config(page_title="STATCAL | 4 Pertanyaan Regresi Logistik", page_icon="📶", layout="wide")
    st.markdown("""
        <style>
        .block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
        div[data-testid="stMetric"] {border:1px solid #dce7ef; border-radius:12px;
                                      padding:13px; background:#f5f9fc;}
        h1 {font-size:2rem !important;}
        </style>
    """, unsafe_allow_html=True)
    st.title("📶 STATCAL ONLINE — Customer Churn Prediction")
    st.markdown("**Empat Pertanyaan: Prediksi Pelanggan Berhenti Berlangganan (Customer Churn)**")
    st.caption("Kasus → Dataset → Aplikasi → Evaluasi | Bab 2 — Machine Learning Berbasis Studi Kasus")
    st.info("**Studi kasus Bab 2:** Prediksi pelanggan layanan internet yang berhenti berlangganan pada Oktober 2026. Dataset bawaan berisi **1.000 pelanggan sintetis**; hasilnya hanya untuk pembelajaran, bukan keputusan bisnis otomatis.")

    with st.sidebar:
        st.header("⚙️ Pengaturan Model")
        up = st.file_uploader("Unggah CSV atau Excel (opsional)", type=["csv", "xlsx"],
                              help="Jika tidak mengunggah data, aplikasi memakai dataset bawaan 1.000 pelanggan.")
        try:
            if up:
                if up.name.lower().endswith(".csv"):
                    df = pd.read_csv(up)
                else:
                    excel = pd.ExcelFile(up)
                    preferred = "DATA_PELANGGAN" if "DATA_PELANGGAN" in excel.sheet_names else excel.sheet_names[0]
                    sheet = st.selectbox(
                        "Pilih worksheet data", excel.sheet_names,
                        index=excel.sheet_names.index(preferred),
                        help="File Excel buku memiliki beberapa worksheet; pilih DATA_PELANGGAN.",
                    )
                    df = pd.read_excel(excel, sheet_name=sheet)
                st.success(f"Dataset unggahan: {up.name}")
            else:
                if not DEFAULT_DATA.exists():
                    st.error("Dataset bawaan tidak ditemukan. Silakan unggah file CSV/XLSX.")
                    st.stop()
                df = pd.read_csv(DEFAULT_DATA)
                st.caption("Memakai dataset simulasi 1.000 pelanggan.")
        except Exception as e:
            st.error(f"Dataset tidak dapat dibaca: {e}")
            st.stop()
        if df.empty or len(df.columns) < 2:
            st.error("Dataset harus mempunyai baris dan setidaknya dua kolom.")
            st.stop()
        target_options = list(df.columns)
        target = st.selectbox("Target klasifikasi (Y)", target_options,
                              index=target_options.index(DEFAULT_TARGET) if DEFAULT_TARGET in target_options else len(target_options)-1)
        available_classes = sorted(df[target].dropna().astype(str).unique().tolist())
        if len(available_classes) != 2:
            st.error("Target Y harus mempunyai tepat dua kelas.")
            st.stop()
        pos = st.selectbox("Kelas positif (nilai 1)", available_classes,
                           index=available_classes.index("Berhenti") if "Berhenti" in available_classes else 1,
                           help="Pilih kategori yang ingin ditemukan model. Untuk studi kasus ini: Berhenti.")
        candidates = [col for col in df.columns if col != target and col not in ID_COLUMNS]
        selected = st.multiselect("Variabel prediktor (X)", candidates, default=candidates,
                                  help="Pilih karakteristik berlangganan; ID pelanggan tidak dipakai.")
        if not selected:
            st.warning("Pilih setidaknya satu fitur X.")
            st.stop()
        train_percentage = st.slider("Persentase data training", 60, 90, 80, 5)
        threshold = st.slider("Threshold klasifikasi", 0.10, 0.90, 0.50, 0.05,
                              help="Probabilitas ≥ threshold akan diprediksi sebagai kelas positif.")
        with st.expander("Pengaturan lanjutan"):
            C = st.select_slider("Regularisasi (C)", options=[0.01, 0.1, 1.0, 10.0, 100.0], value=1.0)
            balanced = st.checkbox("Seimbangkan bobot kelas", value=False)
            seed = st.number_input("Random seed", min_value=0, max_value=999999, value=42, step=1)
        st.caption("Standar Bab 2: training 80%, testing 20%, threshold 0,50, C=1, bobot kelas tidak aktif, seed 42, kelas positif **Berhenti**.")

    try:
        res = run_logistic_analysis(df, target, pos, selected,
                                    train_percentage, threshold, C, balanced, int(seed))
    except (ValueError, TypeError) as error:
        st.error(f"Model tidak dapat diproses: {error}")
        st.stop()
    except Exception as error:
        st.error(f"Pemodelan gagal. Periksa tipe dan isi data. Detail: {error}")
        st.stop()

    met, neg = res["metrics"], res["negative"]
    # Penanda konfigurasi untuk menghindari penggunaan profil lama setelah model diubah.
    dataset_id = int(pd.util.hash_pandas_object(res["df"], index=True).sum())
    signature = (dataset_id, target, pos, tuple(selected), train_percentage,
                 round(threshold, 3), C, balanced, int(seed))
    if "profil_terakhir" not in st.session_state:
        st.session_state["profil_terakhir"] = None
    saved = st.session_state["profil_terakhir"]
    profile_result = saved["result"] if saved and saved["signature"] == signature else None

    st.success(f"Model siap: **{met['N Training']} data training** dan **{met['N Testing']} data testing**. "
               f"Fokus evaluasi: **{pos} (kelas positif = 1)**.")

    tab_intro, tab_one, tab_two, tab_three, tab_four, tab_excel = st.tabs([
        "📚 Kasus & Data",
        "✅ Q1 • Benar (TP)",
        "⚠️ Q2 • Terlewat (FN)",
        "🎯 Q3 • Recall",
        "🧑‍📶 Q4 • Probabilitas",
        "📥 Laporan Excel",
    ])

    with tab_intro:
        st.header("Mengenal Kasus, Dataset, dan Pembagian Data")
        st.write("Kita ingin mengenali pelanggan yang berpotensi **Berhenti berlangganan pada Oktober 2026** berdasarkan informasi aktivitas dan layanan yang tercatat sebelum Oktober 2026. Model Logistic Regression memprediksi probabilitas, lalu threshold mengubahnya menjadi kategori.")
        c1, c2, c3 = st.columns(3)
        c1.metric("Jumlah pelanggan", len(res["df"]))
        c2.metric("Data training", met["N Training"])
        c3.metric("Data testing", met["N Testing"])
        st.markdown("**Mengapa dibagi?** Data training dipakai untuk *melatih* model. Data testing belum dipakai dalam pelatihan dan digunakan untuk menghitung keempat jawaban praktikum yang relevan dengan evaluasi model (Q1–Q3).")
        st.markdown("#### Dataset lengkap")
        st.dataframe(res["full"], use_container_width=True, hide_index=True, height=245)
        show1, show2 = st.columns(2)
        with show1:
            st.markdown("#### Data Training")
            st.caption("Observasi yang dipakai saat pelatihan model.")
            st.dataframe(res["training"], use_container_width=True, hide_index=True, height=275)
        with show2:
            st.markdown("#### Data Testing")
            st.caption("Observasi untuk evaluasi, termasuk hasil prediksinya.")
            st.dataframe(res["testing"], use_container_width=True, hide_index=True, height=275)
        with st.expander("Kamus variabel / pengertian fitur"):
            descr = pd.DataFrame([{"Variabel": v, "Keterangan": GUIDE.get(v, "Variabel pengguna")}
                                  for v in res["df"].columns])
            st.dataframe(descr, use_container_width=True, hide_index=True)
        with st.expander("Ringkasan kinerja tambahan (opsional)"):
            a, b, c = st.columns(3)
            a.metric("Accuracy", fmt_pct(met["Accuracy"]))
            b.metric("Precision", fmt_pct(met["Precision"]))
            c.metric("ROC-AUC", f"{met['ROC-AUC']:.3f}" if met["ROC-AUC"] is not None else "Tidak tersedia")
        st.caption("Kolom Nomor_Baris_Data mencatat posisi baris sesudah baris tanpa target dihapus. Data Training dipakai melatih model; seluruh hasil Q1–Q3 berasal dari Data Testing.")

    with tab_one:
        st.header(f"Pertanyaan 1 — Berapa pelanggan yang diprediksi {pos} dengan benar?")
        st.write(f"**Indikator: True Positive (TP).** TP adalah pelanggan yang **sebenarnya {pos}** dan juga **diprediksi {pos}** oleh model.")
        st.metric("Jawaban Pertanyaan 1 — True Positive", f"{met['TP']} pelanggan")
        st.success(f"**Interpretasi:** Dari {met['N Testing']} pelanggan pada data testing, terdapat **{met['TP']} pelanggan** yang benar-benar {pos} dan berhasil diprediksi sebagai {pos}.")
        fig = confusion_plot(res["cm"], neg, pos)
        st.pyplot(fig)
        plt.close(fig)
        st.caption("Baca sel kanan bawah Confusion Matrix (baris aktual positif, kolom prediksi positif).")
        with st.expander("Lihat daftar observasi TP (jawaban dapat ditelusuri)"):
            tp_rows = res["testing"].loc[res["testing"]["Kategori_Confusion_Matrix"] == "TP"]
            st.dataframe(tp_rows, use_container_width=True, hide_index=True)

    with tab_two:
        st.header(f"Pertanyaan 2 — Berapa pelanggan yang sebenarnya {pos} tetapi keliru diprediksi {neg}?")
        st.write(f"**Indikator: False Negative (FN).** FN adalah pelanggan yang **sebenarnya {pos}**, tetapi model **memprediksi {neg}**. Inilah kasus berisiko yang terlewat oleh model.")
        st.metric("Jawaban Pertanyaan 2 — False Negative", f"{met['FN']} pelanggan")
        st.warning(f"**Interpretasi:** Sebanyak **{met['FN']} pelanggan** dalam data testing sebenarnya {pos}, tetapi tidak ditandai sebagai {pos} oleh model.")
        fig = confusion_plot(res["cm"], neg, pos)
        st.pyplot(fig)
        plt.close(fig)
        st.caption("Baca sel kiri bawah Confusion Matrix (baris aktual positif, kolom prediksi negatif).")
        with st.expander("Lihat daftar pelanggan FN / salah prediksi"):
            fn_rows = res["testing"].loc[res["testing"]["Kategori_Confusion_Matrix"] == "FN"]
            st.dataframe(fn_rows, use_container_width=True, hide_index=True)

    with tab_three:
        st.header(f"Pertanyaan 3 — Seberapa baik model mengenali pelanggan berisiko {pos}?")
        st.markdown("**Indikator: Recall (sensitivitas)**, yaitu persentase dari seluruh pelanggan yang sebenarnya termasuk kelas positif yang berhasil ditemukan model.")
        st.metric("Jawaban Pertanyaan 3 — Recall", fmt_pct(met["Recall"]))
        st.progress(min(1.0, max(0.0, met["Recall"])),
                    text=f"{met['TP']} dari {met['Jumlah Aktual Positif']} pelanggan {pos} berhasil ditemukan")
        st.info(f"**Interpretasi:** Pada data testing, dari **{met['Jumlah Aktual Positif']} pelanggan yang sebenarnya {pos}**, model berhasil menemukan **{met['TP']} pelanggan** ({fmt_pct(met['Recall'])}). Sementara itu, **{met['FN']} pelanggan** lainnya terlewat.")
        with st.expander("Bagaimana angka Recall diperoleh?"):
            st.latex(r"\mathrm{Recall}=\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FN}}")
            st.write(f"Recall = {met['TP']} ÷ ({met['TP']} + {met['FN']}) = **{fmt_pct(met['Recall'])}**.")
        st.caption("Recall bukan accuracy keseluruhan. Recall fokus pada kemampuan model menemukan kelas positif.")

    with tab_four:
        st.header(f"Pertanyaan 4 — Berapa probabilitas seorang pelanggan diprediksi {pos}?")
        st.write("Masukkan **profil pelanggan contoh**. Model menghitung probabilitas Berhenti berdasarkan data hingga September 2026. Profil bersifat simulasi, bukan identitas pelanggan sungguhan.")
        with st.form("form_profil_pelanggan"):
            entry = {}
            col1, col2 = st.columns(2)
            for i, feature in enumerate(selected):
                with (col1 if i % 2 == 0 else col2):
                    if feature in res["numeric"]:
                        vals = pd.to_numeric(res["X_train"][feature], errors="coerce")
                        default_val = float(vals.median()) if vals.notna().any() else 0.0
                        if feature == "Tagihan_Bulanan":
                            entry[feature] = st.number_input(feature, min_value=0.0, value=float(round(default_val, -3)), step=10000.0, format="%.0f", help=GUIDE.get(feature, "Masukkan angka dalam rupiah."))
                        elif feature == "Kepuasan_Pelanggan":
                            entry[feature] = st.number_input(feature, min_value=1, max_value=5, value=int(round(default_val)), step=1, help=GUIDE.get(feature, "Skala 1 sampai 5."))
                        elif feature in ("Jumlah_Keluhan", "Keterlambatan_Pembayaran", "Gangguan_Layanan", "Frekuensi_Penggunaan", "Lama_Berlangganan_Bulan"):
                            entry[feature] = st.number_input(feature, min_value=0, value=max(0,int(round(default_val))), step=1, help=GUIDE.get(feature, "Masukkan angka."))
                        else:
                            entry[feature] = st.number_input(feature, min_value=0.0, value=max(0.,round(default_val, 2)), step=1.0, help=GUIDE.get(feature, "Masukkan angka."))
                    else:
                        values = sorted(res["X_train"][feature].dropna().astype(str).unique().tolist())
                        entry[feature] = st.selectbox(feature, values,
                                                       help=GUIDE.get(feature, "Pilih kategori."))
            submit = st.form_submit_button("🔎 Hitung Probabilitas Profil", type="primary")
        if submit:
            try:
                profile_result = predict_customer_profile(res, entry)
                st.session_state["profil_terakhir"] = {"signature": signature, "result": profile_result}
            except Exception as e:
                st.error(f"Simulasi gagal: {e}")
        if profile_result:
            c1, c2 = st.columns(2)
            c1.metric(f"Probabilitas {pos}", fmt_pct(profile_result["probability"]))
            c2.metric("Kelas prediksi", profile_result["prediction"])
            st.progress(profile_result["probability"], text=f"Probabilitas kelas positif: {fmt_pct(profile_result['probability'])}")
            st.success(f"**Interpretasi:** Berdasarkan profil contoh ini, model memperkirakan probabilitas **{pos} sebesar {fmt_pct(profile_result['probability'])}**. Pada threshold {threshold:.2f}, kelas hasil prediksi adalah **{profile_result['prediction']}**.")
            st.caption("Hasil simulasi profil ini otomatis akan ikut disimpan dalam file laporan Excel.")
        else:
            st.info("Klik **Hitung Probabilitas Profil** terlebih dahulu. Setelah dijalankan, hasil akan muncul di sini dan di laporan Excel.")
        st.caption("Probabilitas model bukan kepastian pelanggan akan berhenti. Mengubah satu nilai profil tidak membuktikan hubungan sebab-akibat.")

    with tab_excel:
        st.header("Unduh Seluruh Hasil dalam Satu File Excel")
        st.write("File Excel dibuat ulang berdasarkan **pengaturan model yang sedang aktif**. Dengan demikian, jawaban pertanyaan, pembagian data, dan prediksi testing berasal dari model yang sama.")
        summary = pd.DataFrame([
            ("Q1 — TP", met["TP"]),
            ("Q2 — FN", met["FN"]),
            ("Q3 — Recall", fmt_pct(met["Recall"])),
            ("Q4 — Probabilitas profil", fmt_pct(profile_result["probability"]) if profile_result else "Belum dijalankan"),
        ], columns=["Pertanyaan", "Hasil"])
        st.dataframe(summary, hide_index=True, use_container_width=True)
        st.markdown("#### Lihat kembali data training dan testing")
        select_view = st.radio("Tabel yang ditampilkan", ["Data Testing + Prediksi", "Data Training", "Prediksi Testing (ringkas)"], horizontal=True)
        if select_view == "Data Training":
            shown = res["training"]
        elif select_view == "Prediksi Testing (ringkas)":
            shown = res["prediction"]
        else:
            shown = res["testing"]
        st.dataframe(shown, use_container_width=True, hide_index=True, height=280)
        with st.expander("Apa saja isi workbook Excel yang diunduh?"):
            st.markdown("""
            **10 worksheet:** Ringkasan; Pertanyaan & Jawaban; Pengaturan Model;
            Data Lengkap; Data Training; Data Testing; Prediksi Testing;
            Metrik Evaluasi; Simulasi Profil; Kamus Variabel; ditambah
            **Confusion Matrix** (total 11 sheet).

            *Data Training* menampilkan observasi yang digunakan melatih model.
            *Data Testing* menampilkan observasi yang digunakan untuk mengevaluasi model,
            beserta probabilitas, kelas hasil prediksi, benar/salah, dan kategori TP/FN/FP/TN.
            """)
        if not profile_result:
            st.warning("Q4 belum diisi. Jika ingin probabilitas profil ikut tercetak di Excel, jalankan dahulu simulasi pada Tab 4.")
        try:
            excel_bytes = create_excel_report(res, profile_result)
            st.download_button("📥 Download Semua Hasil — Excel (.xlsx)",
                               data=excel_bytes,
                               file_name="STATCAL_Bab2_Customer_Churn_Hasil.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                               type="primary", use_container_width=True)
        except Exception as error:
            st.error(f"Tidak berhasil membuat Excel: {error}")
        left, right = st.columns(2)
        with left:
            st.download_button("Download Data Training (CSV)",
                               res["training"].to_csv(index=False).encode("utf-8-sig"),
                               file_name="Data_Training.csv", mime="text/csv", use_container_width=True)
        with right:
            st.download_button("Download Data Testing (CSV)",
                               res["testing"].to_csv(index=False).encode("utf-8-sig"),
                               file_name="Data_Testing_dengan_Prediksi.csv", mime="text/csv", use_container_width=True)
        st.caption("Hasil berasal dari satu pembagian data terstratifikasi; gunakan sebagai latihan, bukan keputusan pelanggan secara otomatis.")

    st.divider()
    st.caption("STATCAL ONLINE • Machine Learning Lab • Bab 2 Customer Churn • Regresi Logistik • Data sintetis")


if __name__ == "__main__":
    start_app()
