import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix
import talib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# --- Page Configuration ---
st.set_page_config(page_title="Advanced Stock Prediction Engine", layout="wide")

#==============================================================================
# 1. DATA PROCESSING FUNCTIONS
#==============================================================================

@st.cache_data
def fetch_data(ticker, start, end):
    """Fetches historical stock data from Yahoo Finance with error handling and normalized OHLCV columns."""
    try:
        df = yf.download(ticker, start=start, end=end)
        if df is None or df.empty:
            return None

        # Normalize columns (yfinance can return MultiIndex or different names)
        if isinstance(df.columns, pd.MultiIndex):
            # If user passed a single ticker but got multiindex, take first level names
            # e.g., ('Open',) -> 'Open' or ('Adj Close',) -> 'Adj Close'
            df.columns = df.columns.get_level_values(0)

        # Unify common variations
        rename_map = {
            "Adj Close": "Adj Close",
            "adjclose": "Adj Close",
            "close": "Close",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "volume": "Volume",
        }
        df = df.rename(columns=rename_map)

        # If Close missing but Adj Close present, use Adj Close as Close
        if "Close" not in df.columns and "Adj Close" in df.columns:
            df["Close"] = df["Adj Close"]

        required = ["Open", "High", "Low", "Close", "Volume"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            # Some feeds miss Volume for indices/crypto; try to fill minimal viable defaults
            if "Volume" in missing:
                df["Volume"] = 0.0
                missing = [c for c in missing if c != "Volume"]
            if missing:  # still missing critical OHLC
                return None

        # Ensure numeric dtypes
        for c in ["Open", "High", "Low", "Close", "Volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["Open", "High", "Low", "Close"])

        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

def calculate_features(df):
    """Calculates technical indicators using TA-Lib with additional features."""
    try:
        # Ensure required columns exist
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
            return df

        # Convert to numpy arrays (float64)
        close_prices = np.asarray(df['Close'], dtype=np.float64).flatten()
        high_prices  = np.asarray(df['High'],  dtype=np.float64).flatten()
        low_prices   = np.asarray(df['Low'],   dtype=np.float64).flatten()
        open_prices  = np.asarray(df['Open'],  dtype=np.float64).flatten()
        volume       = np.asarray(df['Volume'],dtype=np.float64).flatten()

        if len(close_prices) < 50:
            st.warning("Insufficient data for technical indicators. Some features may be missing.")
            return df

        # Technical indicators
        df['RSI'] = talib.RSI(close_prices, timeperiod=14)

        macd_result = talib.MACD(close_prices, fastperiod=12, slowperiod=26, signalperiod=9)
        df['MACD'] = macd_result[0]
        df['MACD_signal'] = macd_result[1]
        df['MACD_hist'] = macd_result[2]

        bb_result = talib.BBANDS(close_prices, timeperiod=20)
        df['Upper_Band'] = bb_result[0]
        df['Middle_Band'] = bb_result[1]
        df['Lower_Band'] = bb_result[2]

        df['SMA_10'] = talib.SMA(close_prices, timeperiod=10)
        df['SMA_50'] = talib.SMA(close_prices, timeperiod=50)
        df['EMA_12'] = talib.EMA(close_prices, timeperiod=12)
        df['ATR'] = talib.ATR(high_prices, low_prices, close_prices, timeperiod=14)
        df['CCI'] = talib.CCI(high_prices, low_prices, close_prices, timeperiod=14)
        df['Williams_R'] = talib.WILLR(high_prices, low_prices, close_prices, timeperiod=14)

        # Price-based features
        df['Price_Change'] = df['Close'].pct_change()
        df['Volume_Change'] = df['Volume'].pct_change()
        df['High_Low_Ratio'] = df['High'] / df['Low']
        df['Close_Open_Ratio'] = df['Close'] / df['Open']

        # Volatility features
        df['Volatility_5'] = df['Price_Change'].rolling(window=5).std()
        df['Volatility_20'] = df['Price_Change'].rolling(window=20).std()

        df.dropna(inplace=True)

        st.success(f"✅ Successfully calculated {len([col for col in df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']])} technical indicators")
        return df

    except Exception as e:
        st.error(f"Error calculating features: {e}")
        st.info("Falling back to basic features only...")
        try:
            close_prices = np.asarray(df['Close'], dtype=np.float64).flatten()
            df['RSI'] = talib.RSI(close_prices, timeperiod=14)
            df['SMA_10'] = talib.SMA(close_prices, timeperiod=10)
            df['Price_Change'] = df['Close'].pct_change()
            df.dropna(inplace=True)
            return df
        except Exception:
            return df

def prepare_data_for_lstm(df):
    """Prepares data for LSTM with enhanced feature set."""
    try:
        # Target variable
        df['Direction'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        # NOTE: keep the last NaN row (from shift) until after scaling alignment
        # We'll align Direction by index later.

        potential_features = [
            'Open','High','Low','Close','Volume','RSI','MACD','MACD_signal',
            'MACD_hist','Upper_Band','Middle_Band','Lower_Band','SMA_10','SMA_50',
            'EMA_12','ATR','CCI','Williams_R','Price_Change','Volume_Change',
            'High_Low_Ratio','Close_Open_Ratio','Volatility_5','Volatility_20'
        ]
        features_to_scale = [f for f in potential_features if f in df.columns]

        if len(features_to_scale) < 5:
            st.warning(f"Only {len(features_to_scale)} features available. Model performance may be limited.")

        scalers = {}
        scaled_feature_data = pd.DataFrame(index=df.index)

        for feature in features_to_scale:
            try:
                feat_series = df[feature].replace([np.inf, -np.inf], np.nan)
                scaler = MinMaxScaler(feature_range=(0, 1))
                valid = feat_series.dropna()
                if len(valid) == 0:
                    continue
                scaled_vals = scaler.fit_transform(valid.values.reshape(-1,1)).flatten()
                scaled_feature_data.loc[valid.index, feature + '_scaled'] = scaled_vals
                scalers[feature] = scaler
            except Exception as e:
                st.warning(f"Skipping feature {feature} due to scaling error: {e}")

        scaled_feature_data.dropna(inplace=True)

        # Align Direction to the scaled features index, then drop any remaining NaN
        direction_aligned = df.loc[scaled_feature_data.index, 'Direction'].dropna()

        # Ensure the feature frame and target share the same index
        common_idx = scaled_feature_data.index.intersection(direction_aligned.index)
        scaled_feature_data = scaled_feature_data.loc[common_idx]
        direction_aligned = direction_aligned.loc[common_idx].astype(int)

        st.info(f"✅ Prepared {len(features_to_scale)} features for LSTM training with {len(scaled_feature_data)} samples")

        return scaled_feature_data, direction_aligned, scalers

    except Exception as e:
        st.error(f"Error preparing data for LSTM: {e}")
        # Minimal fallback
        scaled_data = pd.DataFrame(index=df.index)
        if 'Close' not in df.columns:
            raise ValueError("Close column missing; cannot create fallback features.")
        scaler = MinMaxScaler()
        scaled_data['Close_scaled'] = scaler.fit_transform(df[['Close']])
        direction = (df['Close'].shift(-1) > df['Close']).astype(int).dropna()
        common_idx = scaled_data.index.intersection(direction.index)
        return scaled_data.loc[common_idx], direction.loc[common_idx], {'Close': scaler}

#==============================================================================
# 2. MODEL TRAINING FUNCTION
#==============================================================================

@st.cache_resource
def train_classifier_model(scaled_feature_data, direction_target, lookback_period=60):
    """Enhanced LSTM model with better architecture."""
    X_clf, y_clf = [], []
    scaled_data = scaled_feature_data.values

    for i in range(lookback_period, len(scaled_data)):
        if i < len(direction_target):
            X_clf.append(scaled_data[i-lookback_period:i])
            y_clf.append(direction_target.iloc[i])

    X_clf, y_clf = np.array(X_clf), np.array(y_clf)

    # Guard against insufficient data
    if len(X_clf) < 100:
        raise ValueError("Not enough sequences after lookback to train the model. Try earlier start date or shorter lookback.")

    split_idx = int(len(X_clf) * 0.8)
    X_train, X_val = X_clf[:split_idx], X_clf[split_idx:]
    y_train, y_val = y_clf[:split_idx], y_clf[split_idx:]

    # Class weights
    classes = np.unique(y_train)
    cw = class_weight.compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weights_dict = {int(k): float(v) for k, v in zip(classes, cw)}

    # Model
    clf_model = Sequential([
        LSTM(units=100, return_sequences=True, input_shape=(X_clf.shape[1], X_clf.shape[2])),
        BatchNormalization(),
        Dropout(0.3),
        LSTM(units=50, return_sequences=True),
        BatchNormalization(),
        Dropout(0.3),
        LSTM(units=25, return_sequences=False),
        BatchNormalization(),
        Dropout(0.2),
        Dense(units=25, activation='relu'),
        Dropout(0.2),
        Dense(units=1, activation='sigmoid')
    ])

    clf_model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall")
        ]
    )

    callbacks = [
        EarlyStopping(monitor='val_accuracy', mode='max', patience=15, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-4)
    ]

    history = clf_model.fit(
        X_train, y_train,
        epochs=150,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        class_weight=class_weights_dict,
        verbose=0
    )

    y_pred_prob = clf_model.predict(X_val, verbose=0)
    y_pred = (y_pred_prob > 0.5).astype(int)

    return clf_model, history, (y_val, y_pred, y_pred_prob)

#==============================================================================
# 3. PLOTTING FUNCTIONS
#==============================================================================

def plot_price_history(df, ticker):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'],
        name='Price'
    ))
    if 'Upper_Band' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['Upper_Band'], name='Upper Band',
                                 line=dict(color='rgba(255,0,0,0.3)')))
        fig.add_trace(go.Scatter(x=df.index, y=df['Lower_Band'], name='Lower Band',
                                 line=dict(color='rgba(255,0,0,0.3)')))
    if 'SMA_50' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], name='SMA 50', line=dict(color='blue')))
    fig.update_layout(title=f"{ticker} Price History with Technical Indicators",
                      xaxis_rangeslider_visible=False, height=600)
    return fig

def plot_training_metrics(history):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes[0,0].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0,0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0,0].set_title('Model Accuracy'); axes[0,0].set_xlabel('Epoch'); axes[0,0].set_ylabel('Accuracy'); axes[0,0].legend()

    axes[0,1].plot(history.history['loss'], label='Training Loss')
    axes[0,1].plot(history.history['val_loss'], label='Validation Loss')
    axes[0,1].set_title('Model Loss'); axes[0,1].set_xlabel('Epoch'); axes[0,1].set_ylabel('Loss'); axes[0,1].legend()

    if 'precision' in history.history:
        axes[1,0].plot(history.history['precision'], label='Training Precision')
        axes[1,0].plot(history.history['val_precision'], label='Validation Precision')
        axes[1,0].set_title('Model Precision'); axes[1,0].set_xlabel('Epoch'); axes[1,0].set_ylabel('Precision'); axes[1,0].legend()

    if 'recall' in history.history:
        axes[1,1].plot(history.history['recall'], label='Training Recall')
        axes[1,1].plot(history.history['val_recall'], label='Validation Recall')
        axes[1,1].set_title('Model Recall'); axes[1,1].set_xlabel('Epoch'); axes[1,1].set_ylabel('Recall'); axes[1,1].legend()

    plt.tight_layout()
    return fig

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'])
    plt.title('Confusion Matrix'); plt.ylabel('Actual'); plt.xlabel('Predicted')
    return fig

#==============================================================================
# 4. STREAMLIT APP
#==============================================================================

# ==============================================================================
# 4. STREAMLIT APP (ENHANCED UI)
# ==============================================================================

# --- Page Configuration ---
st.set_page_config(
    page_title="Stock Prediction Engine",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Main Title ---
st.title("🚀 Advanced Stock Prediction Engine")
st.markdown("""
Train a sophisticated **LSTM Neural Network** with advanced technical indicators.
Get real-time predictions on stock market direction with confidence scoring. 📊
""")
st.divider()

# --- Sidebar for User Inputs ---
with st.sidebar:
    st.header("⚙️ Model Configuration")
    
    # Use a form for a cleaner look and single-submit action
    with st.form(key='settings_form'):
        ticker = st.text_input("Stock Ticker", "TSLA").upper()
        
        min_data_years = st.slider(
            "Training Data Span (Years)", 
            min_value=1, max_value=10, value=3,
            help="Number of years of historical data to use for training."
        )
        
        start_date = st.date_input(
            "Training Start Date",
            pd.to_datetime('today') - pd.DateOffset(years=int(min_data_years))
        )
        
        lookback_period = st.slider(
            "Lookback Period (Days)", 
            min_value=30, max_value=120, value=60,
            help="Number of previous days' data the model uses to make a prediction."
        )
        
        # The submit button for the form
        run_button = st.form_submit_button(label="🎯 Train Model & Predict", use_container_width=True)

# --- Main App Logic ---
if run_button:
    try:
        progress_bar = st.progress(0, text="Initializing...")
        
        # 1. Data Fetching
        progress_bar.progress(10, text=f"📊 Fetching historical data for {ticker}...")
        raw_df = fetch_data(ticker, start_date, pd.to_datetime('today'))
        
        if raw_df is None or raw_df.empty or len(raw_df) < lookback_period + 50:
            st.error(f"❌ Insufficient data for {ticker}. Please select an earlier start date or a different ticker.")
            progress_bar.empty()
        else:
            # 2. Feature Engineering
            progress_bar.progress(30, text="🔧 Engineering features from market data...")
            featured_df = calculate_features(raw_df.copy())
            scaled_features, direction, scalers = prepare_data_for_lstm(featured_df.copy())

            # 3. Model Training
            progress_bar.progress(60, text="🧠 Training the LSTM Neural Network...")
            # Note: Ensure train_classifier_model does NOT have X_train in its return signature
            # if you are not using the SHAP feature importance plot.
            model, history, eval_data = train_classifier_model(
                scaled_features, direction, lookback_period
            )

            # 4. Prediction
            progress_bar.progress(90, text="🔮 Generating tomorrow's prediction...")
            last_sequence = scaled_features.values[-lookback_period:]
            X_pred = np.array([last_sequence])
            prediction_prob = model.predict(X_pred, verbose=0)[0, 0]
            
            # --- Prepare Final Results ---
            last_close = raw_df['Close'].iloc[-1]
            predicted_direction = "⬆️ UP" if prediction_prob > 0.5 else "⬇️ DOWN"
            confidence = prediction_prob if prediction_prob > 0.5 else 1 - prediction_prob
            val_accuracy = float(np.max(history.history['val_accuracy']))
            
            progress_bar.progress(100, text="✅ Analysis Complete!")
            progress_bar.empty()

            # --- Display Results ---
            st.subheader(f"Results for {ticker}")
            
            # KPI Metrics in a styled container
            with st.container(border=True):
                kpi1, kpi2, kpi3, kpi4 = st.columns(4)
                kpi1.metric("💲 Last Close Price", f"${last_close:,.2f}")
                kpi2.metric("📈 Predicted Direction", predicted_direction)
                kpi3.metric("🎯 Confidence Score", f"{confidence:.2%}")
                kpi4.metric("✅ Model Accuracy", f"{val_accuracy:.2%}")

            # Tabs for detailed analysis
            tab1, tab2, tab3 = st.tabs(["📈 Training Performance", "📊 Historical Data", "💡 Model Insights"])

            with tab1:
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Learning Curves")
                    training_fig = plot_training_metrics(history)
                    st.pyplot(training_fig, clear_figure=True)
                with col2:
                    st.subheader("Confusion Matrix")
                    y_true, y_pred, _ = eval_data
                    cm_fig = plot_confusion_matrix(y_true, y_pred)
                    st.pyplot(cm_fig, clear_figure=True)
                
                    st.subheader("Classification Report")

    # Convert classification report to a DataFrame
                report_dict = classification_report(
        y_true, y_pred, target_names=['Down', 'Up'], output_dict=True
    )
                report_df = pd.DataFrame(report_dict).transpose()
                report_df = report_df.round(3)

    # Show as a styled DataFrame
                st.dataframe(
        report_df.style.background_gradient(cmap="Blues"),
        use_container_width=True
    )


            with tab2:
                st.subheader(f"Historical Price & Indicators for {ticker}")
                chart_fig = plot_price_history(featured_df, ticker)
                st.plotly_chart(chart_fig, use_container_width=True)

            with tab3:
                st.subheader("Summary & Details")
                st.info(f"""
                **Training Summary:**
                - Total Samples Used: **{len(scaled_features) - lookback_period}** trading days
                - Features Engineered: **{len(scaled_features.columns)}** technical indicators
                - Lookback Period: **{lookback_period}** days
                - Best Validation Accuracy: **{val_accuracy:.2%}**

                **Prediction Details:**
                - Predicted Direction for Next Trading Day: **{predicted_direction}**
                - Model Confidence: **{confidence:.2%}**
                - Suggested Action: **{"🟢 Consider Buying" if prediction_prob > 0.5 else "🔴 Consider Selling"}**
                """)
                
            st.warning("⚠️ **Disclaimer:** This tool is for educational purposes only and does not constitute financial advice. Always conduct your own research before making investment decisions.")

    except Exception as e:
        st.error(f"❌ An unexpected error occurred. Please try again.")

        st.exception(e) # This will print the full traceback for debugging
