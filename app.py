import streamlit as st
import yfinance as yf
from tensorflow.keras.models import load_model
import numpy as np
if "halaman" not in st.session_state:
    st.session_state["halaman"] = "beranda"
if 'mulai' not in st.session_state:
    st.session_state["mulai"] = False
if not st.session_state["mulai"]:
    st.sidebar.markdown(
        "<div style='height:56px; width:100%;'></div>",
        unsafe_allow_html=True
    )    
if st.session_state["mulai"]:
    if st.sidebar.button("🏠 Beranda"):
        st.session_state["halaman"] = "beranda"
        st.session_state["mulai"] = False
        st.rerun()
kode = st.sidebar.radio("**Pilih Kode Saham**", options=["BRPT", "BUMI", "EMTK"], index=0)
kode_jk = kode + ".JK"
data_download = yf.download(kode_jk, start='2014-05-06', end='2025-05-06', auto_adjust=False)
data_terbaru = yf.download(kode_jk, period='5y', auto_adjust=False)
jumlah_future = st.sidebar.slider("**Periode Prediksi (Hari)**", 30, 365, 365)
if kode == "BRPT":
    nama_emiten = "PT Barito Pacific Tbk"
    jumlah_neuron = 128
    batch_size = 64
    jumlah_epoch = 100
    model = load_model("model_brpt.keras")
    loss = np.load("loss_brpt.npy")
if kode == "BUMI":
    nama_emiten = "PT Bumi Resources Tbk"
    jumlah_neuron = 64
    batch_size = 16
    jumlah_epoch = 100
    model = load_model("model_bumi.keras")
    loss = np.load("loss_bumi.npy")
if kode == "EMTK":
    nama_emiten = "PT Elang Mahkota Teknologi Tbk"
    jumlah_neuron = 128
    batch_size = 32
    jumlah_epoch = 100
    model = load_model("model_emtk.keras")
    loss = np.load("loss_emtk.npy")
    
if not st.session_state["mulai"]:
    if st.sidebar.button("Mulai 🡆", key="tombol_mulai"):
        st.session_state["mulai"] = True    
        st.session_state["halaman"] = "mulai"
        st.rerun()
if st.session_state["mulai"]:
    st.sidebar.info("Pengaturan di atas silakan disesuaikan dengan kebutuhan. Sistem akan langsung memproses setiap perubahan yang dilakukan.")

if st.session_state["halaman"] == "beranda":
    st.write(":red[**BERANDA**]")
    st.title("Prediksi Harga Saham BRPT, BUMI, dan EMTK Menggunakan LSTM")
    with st.expander("**Tentang Website**"):
        st.markdown(
            "<hr style='border:1px solid #ccc; margin: 0; padding: 0; width: 100%;'>",
            unsafe_allow_html=True
        )
        st.write('''Website ini bertujuan untuk mengimplementasikan dalam dunia nyata model yang telah dilatih pada penelitian "Prediksi Harga Saham Tiga Emiten di BEI dengan Kepemilikan Efek Ekuitas Terbanyak Investor Ritel Domestik Menggunakan LSTM" yang memprediksi harga saham PT Barito Pacific Tbk (BRPT), PT Bumi Resources Tbk (BUMI), dan PT Elang Mahkota Teknologi Tbk (EMTK) menggunakan model Long Short-Term Memory (LSTM).''')
        st.write("Fitur Utama:  \n1.\nOpsi untuk memilih kode saham dan periode prediksi sesuai kebutuhan.  \n2.\nHasil prediksi harga saham berupa grafik dan tabel.  \n3.\nInformasi terkait prediksi harga saham tertinggi dan terendah selama periode prediksi.")
        st.write("Library yang digunakan:  \n1.\nStreamlit  \n2.\nYfinance  \n3.\nNumPy  \n4.\nPandas  \n5.\nScikit-learn  \n6.\nTensorFlow  \n7.\nPlotly")
    with st.expander("**Cara Penggunaan**"):
        st.markdown(
            "<hr style='border:1px solid #ccc; margin: 0; padding: 0; width: 100%;'>",
            unsafe_allow_html=True
        )
        st.write('''Langkah-langkah penggunaan website ini adalah sebagai berikut:  \n1.\nPastikan pengaturan di sidebar sesuai dengan kebutuhan.  \n2.\nTekan tombol "Mulai" pada sidebar.  \n3.\nHasil prediksi akan segera ditampilkan.''')
        
elif st.session_state["halaman"] == "mulai":
    st.header(f"Prediksi Harga Saham {nama_emiten} Menggunakan LSTM")
    import pandas as pd
    tanggal_download_first = data_download.index[0].strftime('%d-%m-%y')
    tanggal_terbaru_last = data_terbaru.index[-1].strftime('%d-%m-%y')
    data_download.columns = data_download.columns.droplevel(1)
    data_terbaru.columns = data_terbaru.columns.droplevel(1)
    data_download_relevan = data_download.drop(columns=['Adj Close'])
    data_terbaru_relevan = data_terbaru.drop(columns=['Adj Close'])
    tanggal_full = pd.date_range(start=data_download.index.min(), end=data_download.index.max(), freq='D')
    tanggal_terbaru_full = pd.date_range(start=data_terbaru.index.min(), end=data_terbaru.index.max(), freq='D')
    data_interpolate = data_download_relevan.reindex(tanggal_full).interpolate()
    data_terbaru_interpolate = data_terbaru_relevan.reindex(tanggal_terbaru_full).interpolate()
    data_x = data_interpolate.values
    data_y = data_interpolate['Close'].values  
    data_x_terbaru = data_terbaru_interpolate.values
    data_y_terbaru = data_terbaru_interpolate['Close'].values  
    data_latih_x = data_x[:-2 * 365]
    data_uji_x = data_x_terbaru[365:][-2 * 365:]
    data_prediksi_x = data_x_terbaru[:2 * 365]
    data_latih_y = data_y[2 * 365:-365]
    data_uji_y = data_y_terbaru[-365:]
    data_latih_x_normalisasi = data_x[:-3 * 365]
    data_latih_y_normalisasi = data_y[:-3 * 365]
    from sklearn.preprocessing import MinMaxScaler
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    scaler_x.fit_transform(data_latih_x_normalisasi)
    data_latih_x_normal = scaler_x.transform(data_latih_x)
    data_uji_x_normal = scaler_x.transform(data_uji_x)
    data_prediksi_x_normal = scaler_x.transform(data_prediksi_x)
    scaler_y.fit_transform(data_latih_y_normalisasi.reshape(len(data_latih_y_normalisasi), 1))
    data_latih_y_normal = scaler_y.transform(data_latih_y.reshape(len(data_latih_y), 1))
    data_uji_y_normal = scaler_y.transform(data_uji_y.reshape(len(data_uji_y), 1))
    latih_x = np.array([data_latih_x_normal[i:i + 365] for i in range(len(data_latih_x_normal) - 365)])
    uji_x = np.array([data_uji_x_normal[i:i + 365] for i in range(len(data_uji_x_normal) - 365)])
    prediksi_x = np.array([data_prediksi_x_normal[i:i + 365] for i in range(len(data_prediksi_x_normal) - 365)])
    latih_y = data_latih_y_normal
    uji_y = data_uji_y_normal
    tanggal_prediksi = pd.date_range((data_terbaru_interpolate.index)[-1] + pd.Timedelta(days=1), periods=jumlah_future)
    prediksi = model.predict(prediksi_x[:jumlah_future])
    prediksi_denormal = scaler_y.inverse_transform(prediksi)
    data_prediksi = pd.DataFrame(prediksi_denormal, columns=['Prediksi'])
    data_prediksi['Tanggal'] = tanggal_prediksi
    data_prediksi.set_index('Tanggal', inplace=True)
    data_ffill = data_terbaru_relevan.reindex(tanggal_terbaru_full, method='ffill')
    data_full_last = data_ffill['Close'].iloc[-1]
    tanggal_full_last = data_ffill.index[-1].strftime('%d-%m-%Y')
    libur_bei = ['2025-05-12', '2025-05-13', '2025-05-29', '2025-05-30', '2025-06-06','2025-06-09',
                 '2025-06-27', '2025-09-05', '2025-12-25', '2025-12-26', '2025-12-31']
    libur_bei = pd.to_datetime(libur_bei)
    libur_weekend = tanggal_prediksi[tanggal_prediksi.weekday >= 5]
    libur_all = libur_bei.union(libur_weekend)
    data_prediksi_adjust = data_prediksi.reset_index()
    data_prediksi_adjust['Tanggal'] = pd.to_datetime(data_prediksi_adjust['Tanggal'])
    data_prediksi_adjust.loc[data_prediksi_adjust['Tanggal'].isin(libur_all), 'Prediksi'] = np.nan

    with st.expander("**📊 Hasil Prediksi**", expanded=True):
        st.markdown(
            "<hr style='border:1px solid #ccc; margin: 0; padding: 0; width: 100%;'>",
            unsafe_allow_html=True
        )
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            data_prediksi_max = data_prediksi_adjust['Prediksi'].max()
            tanggal_prediksi_max = data_prediksi_adjust[data_prediksi_adjust['Prediksi'] == data_prediksi_max]['Tanggal'].iloc[0]
            tanggal_prediksi_max = tanggal_prediksi_max.strftime('%d-%m-%y')
            selisih_max = data_prediksi_max - data_full_last
            st.metric(label=":blue[Harga Tertinggi (Rp)]", value=f"{data_prediksi_max:.7g}", delta=f"{selisih_max:.5g} ({tanggal_prediksi_max})", help=f"Selisih: Apabila dibandingkan dengan data historis terakhir ({tanggal_full_last}).")
        with col2:
            data_prediksi_min = data_prediksi_adjust['Prediksi'].min()
            tanggal_prediksi_min = data_prediksi_adjust[data_prediksi_adjust['Prediksi'] == data_prediksi_min]['Tanggal'].iloc[0]
            tanggal_prediksi_min = tanggal_prediksi_min.strftime('%d-%m-%y')
            selisih_min = data_prediksi_min - data_full_last
            st.metric(label=":blue[Harga Terendah (Rp)]", value=f"{data_prediksi_min:.7g}", delta=f"{selisih_min:.5g} ({tanggal_prediksi_min})", help=f"Selisih: Apabila dibandingkan dengan data historis terakhir ({tanggal_full_last}).")
        with col3:
            data_prediksi_mean = data_prediksi_adjust['Prediksi'].mean()
            selisih_mean = data_prediksi_mean - data_full_last
            st.metric(label=":blue[Harga Rata-Rata (Rp)]", value=f"{data_prediksi_mean:.7g}", delta=f"{selisih_mean:.5g}", help=f"Selisih: Apabila dibandingkan dengan data historis terakhir ({tanggal_full_last}).")
        with col4:
            st.metric(label=":blue[Periode (Hari)]", value=jumlah_future)
        st.write(" ")
        st.write("**Grafik Hasil Prediksi:**")
        data_prediksi_first = data_prediksi_adjust['Prediksi'].iloc[0]
        if pd.isna(data_prediksi_first):  
            data_prediksi_adjust.at[data_prediksi_adjust.index[0], 'Prediksi'] = data_full_last
        data_prediksi_adjust['Prediksi'] = data_prediksi_adjust['Prediksi'].ffill()
        data_prediksi_adjust.set_index('Tanggal', inplace=True)
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data_ffill.index[-3 * 365:], 
            y=data_ffill['Close'][-3 * 365:], 
            mode='lines', 
            name='Data Historis'
        ))
        fig.add_trace(go.Scatter(
            x=[data_ffill.index[-1], data_prediksi_adjust.index[0]],
            y=[data_ffill['Close'].iloc[-1], data_prediksi_adjust['Prediksi'].iloc[0]],
            mode='lines',
            line_color='red',
            showlegend=False,
            hoverinfo='skip'
        ))
        fig.add_trace(go.Scatter(
            x=data_prediksi_adjust.index, 
            y=data_prediksi_adjust['Prediksi'], 
            mode='lines', 
            name='Prediksi'
        ))
        fig.update_layout(
            xaxis_title='Waktu',
            yaxis_title='Harga (Rp)',
            hovermode='x',
            legend_title='Keterangan',
            margin=dict(t=20, b=50)
        )
        st.plotly_chart(fig)
        st.write("**Tabel Hasil Prediksi:**")
        st.markdown(
            f'''
            <div style=padding-left:1px><b style="font-size:13px; color:grey">Keterangan</b></div>
            <div style=padding-left:1px>
            <span style="display:inline-block;width:15px;height:15px;background-color:lightgreen;"></span>
            <span style="font-size:13px; color:grey; vertical-align:top">= Lebih tinggi dari data historis terakhir ({tanggal_full_last})</span>
            </div>
            <div style=padding-left:1px>
            <span style="display:inline-block;width:15px;height:15px;background-color:lightcoral;"></span>
            <span style="font-size:13px; color:grey; vertical-align:top">= Lebih rendah dari data historis terakhir ({tanggal_full_last})</span>
            </div>
            <div style=padding-left:1px>
            <span style="display:inline-block;width:15px;height:15px;background-color:gold;"></span>
            <span style="font-size:13px; color:grey; vertical-align:top">= Harga prediksi tertinggi</span>
            </div>
            <div style=padding-left:1px>
            <span style="display:inline-block;width:15px;height:15px;background-color:lightpink;"></span>
            <span style="font-size:13px; color:grey; vertical-align:top">= Harga prediksi terendah</span>
            </div>
            <div style=padding-left:1px>
            <span style="display:inline-block;width:15px;height:15px;background-color:lightgrey;"></span>
            <span style="font-size:13px; color:grey; vertical-align:top">= Libur BEI</span>
            </div>
            ''', 
            unsafe_allow_html=True
        )
        data_prediksi_adjust = data_prediksi_adjust.reset_index()
        data_prediksi_adjust['Tahun'] = data_prediksi_adjust['Tanggal'].dt.year
        bulan_transform = {
            1: '01(Januari)', 2: '02(Februari)', 3: '03(Maret)', 4: '04(April)',
            5: '05(Mei)', 6: '06(Juni)', 7: '07(Juli)', 8: '08(Agustus)',
            9: '09(September)', 10: '10(Oktober)', 11: '11(November)', 12: '12(Desember)'
        }
        data_prediksi_adjust['Bulan'] = data_prediksi_adjust['Tanggal'].dt.month.map(bulan_transform)
        data_prediksi_adjust['Tanggal'] = data_prediksi_adjust['Tanggal'].dt.day
        tabel_pivot = data_prediksi_adjust.pivot_table(index='Tanggal', columns=['Tahun', 'Bulan'], values='Prediksi', aggfunc='last')
        set_libur_all = {(d.day, bulan_transform[d.month], d.year) for d in libur_all}
        def warna(val, tanggal, bulan, tahun):
            if pd.isna(val):
                return ''
            elif (tanggal, bulan, tahun) in set_libur_all:
                return 'background-color: lightgray'
            elif val == data_prediksi_max:
                return 'background-color: gold'
            elif val == data_prediksi_min:
                return 'background-color: lightpink'
            elif val > data_full_last:
                return 'background-color: lightgreen'
            elif val < data_full_last:
                return 'background-color: lightcoral'
            return ''
        def apply_warna(df):
            def warna_sel(row):
                style = []
                for col in df.columns:
                    tahun, bulan = col
                    bulan_angka = bulan
                    tanggal = row.name
                    style.append(warna(row[col], tanggal, bulan_angka, tahun))
                return style
            return df.style.apply(warna_sel, axis=1).set_table_styles([
                {'selector': 'th',
                'props': [('text-align', 'left')]}
            ])
        pivot_warna = apply_warna(tabel_pivot)
        st.dataframe(pivot_warna)

    with st.expander("**📊 Data Historis**"):
        data_gabungan = pd.concat([data_download, data_terbaru])
        data_gabungan = data_gabungan[~data_gabungan.index.duplicated(keep='last')]
        jumlah_gabungan = int(len(data_gabungan))
        st.markdown(
            "<hr style='border:1px solid #ccc; margin: 0; padding: 0; width: 100%;'>",
            unsafe_allow_html=True
        )
        col1, col2, col3, col4 = st.columns([2,3,3,3])
        with col1:
            st.metric(label=":blue[Data Historis]", value=jumlah_gabungan)
        with col2:
            st.metric(label=":blue[Tanggal Pertama]", value=tanggal_download_first)
        with col3:
            st.metric(label=":blue[Tanggal Terakhir]", value=tanggal_terbaru_last)
        with col4:
            st.metric(label=":blue[Kode Saham]", value=kode)
        st.write(" ")
        st.write("**Tabel Data Historis:**")
        st.dataframe(data_gabungan)
        st.write(" ")
        st.write("**Grafik Data Historis:**")
        fig = go.Figure()
        for kolom in data_gabungan.columns:
            if kolom != 'Volume':
                fig.add_trace(go.Scatter(
                    x=data_gabungan.index, 
                    y=data_gabungan[kolom], 
                    mode='lines', 
                    name=kolom
                ))
        fig.update_layout(
            xaxis_title='Waktu',
            yaxis_title='Harga (Rp)',
            hovermode='x',
            legend_title='Keterangan',
            margin=dict(t=20, b=50)
        )
        st.plotly_chart(fig)
        fig = go.Figure()
        for kolom in data_gabungan.columns:
            if kolom == 'Volume':
                fig.add_trace(go.Scatter(
                    x=data_gabungan.index, 
                    y=data_gabungan[kolom], 
                    mode='lines', 
                    name=kolom,
                    showlegend=True
                ))
        fig.update_layout(
            xaxis_title='Waktu',
            yaxis_title='Jumlah (lembar)',
            hovermode='x',
            legend_title='Keterangan',
            margin=dict(t=20, b=50)
        )
        st.plotly_chart(fig)

    with st.expander("**📊 Preprocessing Data**"):
        data_gabungan_relevan = data_gabungan.drop(columns=['Adj Close'])
        tanggal_gabungan = pd.date_range(start=data_gabungan_relevan.index.min(), end=data_gabungan_relevan.index.max(), freq='D')                
        data_gabungan_interpolate = data_gabungan_relevan.reindex(tanggal_gabungan).interpolate()
        jumlah_gabungan_interpolate = int(len(data_gabungan_interpolate))
        st.markdown(
            "<hr style='border:1px solid #ccc; margin: 0; padding: 0; width: 100%;'>",
            unsafe_allow_html=True
        )
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(label=":blue[Data Preprocessed]", value=jumlah_gabungan_interpolate)
        with col2:
            st.metric(label=":blue[Data Latih]", value=2923)
        with col3:
            st.metric(label=":blue[Data Uji]", value=365)
        with col4:
            st.metric(label=":blue[Variabel Target]", value='Close')
        st.write(" ")
        st.write("**Tabel Setelah Pemilihan Kolom Relevan:**")
        st.dataframe(data_download_relevan)
        st.write(" ")
        st.write("**Tabel Setelah Handling Missing Values:**")
        st.dataframe(data_gabungan_interpolate)
        st.write(" ")
        st.write("**Tabel Data Latih:**")
        col1, col2 = st.columns([2,1])
        with col1:
            st.markdown(
                '''<div style="font-size:15px; margin-bottom:8px">Fitur (x)</div>''',
                unsafe_allow_html=True
            )
            st.dataframe(data_latih_x)
        with col2:
            st.markdown(
                '''<div style="font-size:15px; margin-bottom:8px">Target (y)</div>''',
                unsafe_allow_html=True
            )
            st.dataframe(data_latih_y)
        st.write(" ")
        st.write("**Tabel Data Uji:**")
        col1, col2 = st.columns([2,1])
        with col1:
            st.markdown(
                '''<div style="font-size:15px; margin-bottom:8px">Fitur (x)</div>''',
                unsafe_allow_html=True
            )
            st.dataframe(data_uji_x)
        with col2:
            st.markdown(
                '''<div style="font-size:15px; margin-bottom:8px">Target (y)</div>''',
                unsafe_allow_html=True
            )
            st.dataframe(data_uji_y)
        st.write(" ")
        st.write("**Tabel Data Latih Setelah Normalisasi:**")
        col1, col2 = st.columns([4,1])
        with col1:
            st.markdown(
                '''<div style="font-size:15px; margin-bottom:8px">Fitur (x)</div>''',
                unsafe_allow_html=True
            )
            st.dataframe(data_latih_x_normal)
        with col2:
            st.markdown(
                '''<div style="font-size:15px; margin-bottom:8px">Target (y)</div>''',
                unsafe_allow_html=True
            )
            st.dataframe(data_latih_y_normal)
        st.write(" ")
        st.write("**Tabel Data Uji Setelah Normalisasi:**")
        col1, col2 = st.columns([4,1])
        with col1:
            st.markdown(
                '''<div style="font-size:15px; margin-bottom:8px">Fitur (x)</div>''',
                unsafe_allow_html=True
            )
            st.dataframe(data_uji_x_normal)
        with col2:
            st.markdown(
                '''<div style="font-size:15px; margin-bottom:8px">Target (y)</div>''',
                unsafe_allow_html=True
            )
            st.dataframe(data_uji_y_normal)

    with st.expander("**📊 Pembentukan Model**"):
        st.markdown(
            "<hr style='border:1px solid #ccc; margin: 0; padding: 0; width: 100%;'>",
            unsafe_allow_html=True
        )
        col1, col2, col3, col4 = st.columns([3,2,2,2])
        with col1:
            st.metric(label=":blue[Jenis Model]", value="Sequential")
        with col2:
            st.metric(label=":blue[Jumlah Layer]", value=3)
        with col3:
            st.metric(label=":blue[Optimizer]", value="Adam")
        with col4:
            st.metric(label=":blue[Loss]", value="MSE")
        st.write(" ")
        st.write("**Tabel Informasi Layer Model:**")
        data_layer = {
            "Layer": ["Input", "LSTM", "Dense (Output)"],
            "Tipe": ["Input Layer", "LSTM", "Dense"],
            "Jumlah Neuron": ["-", jumlah_neuron, 1],
        }
        st.dataframe(data_layer)

    with st.expander("**📊 Pelatihan Model**"):
        st.markdown(
            "<hr style='border:1px solid #ccc; margin: 0; padding: 0; width: 100%;'>",
            unsafe_allow_html=True
        )
        col1, col2, col3, col4 = st.columns([2,2,3,3])
        with col1:
            st.metric(label=":blue[Epoch]", value=jumlah_epoch)
        with col2:
            st.metric(label=":blue[Batch Size]", value=batch_size)
        with col3:
            st.metric(label=":blue[Training Split]", value="100%")
        with col4:
            st.metric(label=":blue[Validation Split]", value="0%")
        st.write(" ")
        st.write("**Grafik Kinerja Model:**")
        range_epoch = list(range(1, len(loss) + 1))
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=range_epoch, 
            y=loss, 
            mode='lines', 
            name='Training Loss'
        ))
        fig.update_layout(
            xaxis_title='Epoch',
            yaxis_title='Loss',
            hovermode='x',
            legend_title='Keterangan',
            margin=dict(t=20, b=50)
        )
        st.plotly_chart(fig)
        st.write(" ")
        st.write("**Tabel Kinerja Model:**")
        data_kinerja = {
            "Epoch": range_epoch,
            "Training Loss": loss,
        }
        st.dataframe(data_kinerja)

    with st.expander("**📊 Pengujian dan Evaluasi Model**"):
        st.markdown(
            "<hr style='border:1px solid #ccc; margin: 0; padding: 0; width: 100%;'>",
            unsafe_allow_html=True
        )
        prediksi_uji = model.predict(uji_x)
        prediksi_uji_denormal = scaler_y.inverse_transform(prediksi_uji)
        uji_y_denormal = scaler_y.inverse_transform(uji_y)
        data_ke = list(range(1, 366))
        prediksi_denormal_list = prediksi_uji_denormal.flatten().tolist()
        uji_y_denormal_list = uji_y_denormal.flatten().tolist()
        dataY = {
            "Data ke-": data_ke,
            "Prediksi": prediksi_denormal_list,
            "Aktual": uji_y_denormal_list
        }
        from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
        mse = mean_squared_error(uji_y_denormal, prediksi_uji_denormal)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(uji_y_denormal, prediksi_uji_denormal)
        mape = mean_absolute_percentage_error(uji_y_denormal, prediksi_uji_denormal) * 100
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(label=":blue[MSE]", value=f'{round(mse, 3)}')
        with col2:
            st.metric(label=":blue[RMSE]", value=f'{round(rmse, 3)}')
        with col3:
            st.metric(label=":blue[MAE]", value=f'{round(mae, 3)}')
        with col4:
            st.metric(label=":blue[MAPE]", value=f'{round(mape, 3)}%')
        st.write(" ")
        st.write("**Grafik Perbandingan Hasil Prediksi Uji dengan Nilai Aktual:**")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dataY['Data ke-'],
            y=dataY['Aktual'],
            mode='lines', 
            name='Aktual'
        ))
        fig.add_trace(go.Scatter(
            x=dataY['Data ke-'],
            y=dataY['Prediksi'],
            mode='lines', 
            line_color='red',
            name='Prediksi'
        ))
        fig.update_layout(
            xaxis_title='Data ke-',
            yaxis_title='Harga (Rp)',
            hovermode='x',
            legend_title='Keterangan',
            margin=dict(t=20, b=50)
        )
        st.plotly_chart(fig)
        st.write(" ")
        st.write("**Tabel Perbandingan Hasil Prediksi Uji dengan Nilai Aktual:**")
        st.dataframe(dataY)
