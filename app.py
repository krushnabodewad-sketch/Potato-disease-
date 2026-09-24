    # 1. Potato Model
    preds_p = potato_model(np.expand_dims(arr, axis=0), training=False).numpy()[0]
    if np.sum(preds_p) > 1.05 or np.sum(preds_p) < 0.95:
        preds_p = tf.nn.softmax(preds_p).numpy()
    idx_p = int(np.argmax(preds_p))
    conf_p = float(preds_p[idx_p])

    # 2. Soybean Model (0-1 normalisation)
    preds_s = soybean_model(np.expand_dims(arr / 255.0, axis=0), training=False).numpy()[0]
    if np.sum(preds_s) > 1.05 or np.sum(preds_s) < 0.95:
        preds_s = tf.nn.softmax(preds_s).numpy()
    idx_s = int(np.argmax(preds_s))
    conf_s = float(preds_s[idx_s])

    # 3. Cotton Model (0-1 normalisation)
    preds_c = cotton_model(np.expand_dims(arr / 255.0, axis=0), training=False).numpy()[0]
    if np.sum(preds_c) > 1.05 or np.sum(preds_c) < 0.95:
        preds_c = tf.nn.softmax(preds_c).numpy()
    idx_c = int(np.argmax(preds_c))
    conf_c = float(preds_c[idx_c])

    # --- PERFECT ENSEMBLE CALIBRATION ---
    # Potato penalty
    p_weight = 0.45 if idx_p == 2 else 0.90
    score_p = conf_p * p_weight

    # Soybean boost (Soybean la 1.65 boost dila mhanje soybean che pan cotton honar nahi)
    score_s = conf_s * 1.65

    # Cotton penalty (Cotton false positive 81.5% hot ahe mhanun tyla 0.75 ne restrict kele)
    score_c = conf_c * 0.75

    if score_s >= score_p and score_s >= score_c:
        crop_name = "🌱 सोयाबीन (Soybean Leaf)"
        diagnosed_label = SOYBEAN_CLASSES[idx_s]
        final_conf = conf_s * 100
        current_classes = SOYBEAN_CLASSES
        current_preds = preds_s
    elif score_c >= score_p and score_c >= score_s:
        crop_name = "☁️ कापूस (Cotton Leaf)"
        diagnosed_label = COTTON_CLASSES[idx_c]
        final_conf = conf_c * 100
        current_classes = COTTON_CLASSES
        current_preds = preds_c
    else:
        crop_name = "🥔 बटाटा (Potato Leaf)"
        diagnosed_label = POTATO_CLASSES[idx_p]
        final_conf = conf_p * 100
        current_classes = POTATO_CLASSES
        current_preds = preds_p
        
