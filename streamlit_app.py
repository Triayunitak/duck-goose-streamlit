if uploaded_file is not None:
    try:
        # 1️⃣ Extract 16 fitur
        features = extract_features(uploaded_file)   # shape (16,)

        # 2️⃣ SCALING DULU (WAJIB 16 fitur)
        features_scaled_full = scaler.transform(
            features.reshape(1, -1)
        )  # shape (1, 16)

        # 3️⃣ FEATURE SELECTION SETELAH SCALING
        features_scaled_selected = features_scaled_full[:, selected_idx]
        # shape (1, 13)

        # 4️⃣ Predict
        prediction = model.predict(features_scaled_selected)
        label = encoder.inverse_transform(prediction)

        st.success(f"🎯 Hasil Prediksi: **{label[0]}**")

    except Exception as e:
        st.error("❌ Terjadi kesalahan saat memproses audio.")
        st.exception(e)
