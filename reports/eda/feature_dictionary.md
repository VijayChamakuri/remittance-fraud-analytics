# Feature dictionary

69 model features. Target = `Class` (1 = fraud).

| Feature | Rationale |
|---|---|
| `V1` | Anonymized PCA-style signal from the card network; carries most of the separating power in ULB-style data. |
| `V2` | Anonymized PCA-style signal from the card network; carries most of the separating power in ULB-style data. |
| `V3` | Anonymized PCA-style signal from the card network; carries most of the separating power in ULB-style data. |
| `V4` | Anonymized PCA-style signal from the card network; carries most of the separating power in ULB-style data. |
| `V5` | Anonymized PCA-style signal from the card network; carries most of the separating power in ULB-style data. |
| `V6` | Anonymized PCA-style signal from the card network; carries most of the separating power in ULB-style data. |
| `V7` | Anonymized PCA-style signal from the card network; carries most of the separating power in ULB-style data. |
| `V8` | Anonymized PCA-style signal from the card network; carries most of the separating power in ULB-style data. |
| `V9` | Anonymized PCA-style signal from the card network; carries most of the separating power in ULB-style data. |
| `V10` | Anonymized PCA-style signal from the card network; carries most of the separating power in ULB-style data. |
| `V11` | Anonymized PCA-style signal from the card network; carries most of the separating power in ULB-style data. |
| `V12` | Anonymized PCA-style signal from the card network; carries most of the separating power in ULB-style data. |
| `V13` | Anonymized PCA-style signal from the card network; carries most of the separating power in ULB-style data. |
| `V14` | Anonymized PCA-style signal from the card network; carries most of the separating power in ULB-style data. |
| `V15` | Anonymized PCA-style signal from the card network; carries most of the separating power in ULB-style data. |
| `V16` | Anonymized PCA-style signal from the card network; carries most of the separating power in ULB-style data. |
| `V17` | Anonymized PCA-style signal from the card network; carries most of the separating power in ULB-style data. |
| `V18` | Anonymized PCA-style signal from the card network; carries most of the separating power in ULB-style data. |
| `V19` | Anonymized PCA-style signal from the card network; carries most of the separating power in ULB-style data. |
| `V20` | Anonymized PCA-style signal from the card network; carries most of the separating power in ULB-style data. |
| `V21` | Anonymized PCA-style signal from the card network; carries most of the separating power in ULB-style data. |
| `V22` | Anonymized PCA-style signal from the card network; carries most of the separating power in ULB-style data. |
| `V23` | Anonymized PCA-style signal from the card network; carries most of the separating power in ULB-style data. |
| `V24` | Anonymized PCA-style signal from the card network; carries most of the separating power in ULB-style data. |
| `V25` | Anonymized PCA-style signal from the card network; carries most of the separating power in ULB-style data. |
| `V26` | Anonymized PCA-style signal from the card network; carries most of the separating power in ULB-style data. |
| `V27` | Anonymized PCA-style signal from the card network; carries most of the separating power in ULB-style data. |
| `V28` | Anonymized PCA-style signal from the card network; carries most of the separating power in ULB-style data. |
| `Amount` | Transaction amount ($). Fraud skews to higher amounts. |
| `log_amount` | Log-scaled amount; stabilizes the heavy right tail for linear models. |
| `hour_of_day` | Local hour; fraud over-indexes in off-peak/night hours. |
| `is_night` | Night flag (0-6h): elevated fraud window. |
| `near_threshold` | Amount within 90-100% of the reporting/limit threshold -> classic structuring signal. |
| `is_round_amount` | Round-number amount; weakly associated with scripted/automated fraud. |
| `account_age_days` | Days since the sending account was created; mule accounts are brand new. |
| `device_change` | Session device differs from the card's usual device -> account-takeover signal. |
| `corridor_new` | First time this card sends on this corridor -> stolen-instrument / new-geo risk. |
| `n_clicks_session` | Clickstream depth; rushed ATO sessions have fewer, faster clicks. |
| `session_duration_s` | Session length; very short sessions correlate with scripted fraud. |
| `txn_gap_s` | Seconds since this card's previous txn; small gaps = velocity/burst behavior. |
| `card_txn_rank` | Nth transaction for this card in-window; captures account tenure/activity. |
| `txn_count_1h` | Txns by this card in the trailing hour; velocity is a core fraud signal. |
| `is_new_account` | Account younger than 30 days -> mule / bust-out risk. |
| `log_txn_gap` | Log of inter-transaction gap; compresses the huge range of idle times. |
| `amount_per_click` | Dollars moved per click; high value = low deliberation, typical of ATO cash-out. |
| `amount_zscore_vs_card` | How anomalous this amount is vs the card's own history (z-score); amount anomalies flag stolen instruments & ATO cash-outs. |
| `amount_to_card_mean` | Ratio of amount to the card's historical average spend. |
| `txn_count_24h` | Txns by this card in the trailing 24h -> sustained velocity / structuring bursts. |
| `card_hist_txn_count` | Number of prior transactions seen for this card (entity tenure). |
| `device_change_x_new_corridor` | Interaction: new device AND new corridor together is a strong takeover/stolen signal. |
| `device_type_android_app` | One-hot indicator for device_type == 'device_type_android_app'. |
| `device_type_ios_app` | One-hot indicator for device_type == 'device_type_ios_app'. |
| `device_type_web_desktop` | One-hot indicator for device_type == 'device_type_web_desktop'. |
| `device_type_web_mobile` | One-hot indicator for device_type == 'device_type_web_mobile'. |
| `channel_app` | One-hot indicator for channel == 'channel_app'. |
| `channel_web` | One-hot indicator for channel == 'channel_web'. |
| `merchant_category_airtime_topup` | One-hot indicator for merchant_category == 'merchant_category_airtime_topup'. |
| `merchant_category_bank_deposit` | One-hot indicator for merchant_category == 'merchant_category_bank_deposit'. |
| `merchant_category_bill_pay` | One-hot indicator for merchant_category == 'merchant_category_bill_pay'. |
| `merchant_category_cash_pickup` | One-hot indicator for merchant_category == 'merchant_category_cash_pickup'. |
| `merchant_category_p2p_transfer` | One-hot indicator for merchant_category == 'merchant_category_p2p_transfer'. |
| `corridor_AU->PH` | One-hot indicator for corridor == 'corridor_AU->PH'. |
| `corridor_CA->IN` | One-hot indicator for corridor == 'corridor_CA->IN'. |
| `corridor_UK->NG` | One-hot indicator for corridor == 'corridor_UK->NG'. |
| `corridor_US->GT` | One-hot indicator for corridor == 'corridor_US->GT'. |
| `corridor_US->IN` | One-hot indicator for corridor == 'corridor_US->IN'. |
| `corridor_US->MX` | One-hot indicator for corridor == 'corridor_US->MX'. |
| `corridor_US->NG` | One-hot indicator for corridor == 'corridor_US->NG'. |
| `corridor_US->PH` | One-hot indicator for corridor == 'corridor_US->PH'. |
