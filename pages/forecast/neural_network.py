import torch
import copy
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn as nn

class SlidingWindowDataset(Dataset):
    def __init__(self, df, target_col="target"):
        self.X = df["all_features"].values
        self.y = df[target_col].values
        self.df = df
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        y = torch.log1p(torch.tensor(self.y[idx], dtype=torch.float32))
        return x, y, self.df.iloc[idx]["product_id_final"]
    



class GlobalLSTMRNN(nn.Module):
    def __init__(
        self,
        n_features=11,
        conv_channels=64,
        lstm_hidden=64,
        rnn_hidden=64,
        rnn_act_fun="TANH",
    ):
        super().__init__()

        # 1️⃣ Conv1D: feature extraction
        self.conv1d = nn.Conv1d(
            in_channels=n_features,
            out_channels=conv_channels,
            kernel_size=3,
            padding=1
        )

        # 2️⃣ LSTM: long-term memory
        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=lstm_hidden,
            batch_first=True
        )

        # 3️⃣ Simple RNN: temporal compression
        self.rnn = nn.RNN(
            input_size=lstm_hidden,
            hidden_size=rnn_hidden,
            batch_first=True
        )

        # 4️⃣ The Custom Activation Layer
        # We define it here based on the string passed by the Bayesian Optimizer
        if rnn_act_fun == "SWISH":
            self.act = nn.SiLU()  # PyTorch's official name for Swish
        elif rnn_act_fun == "GELU":
            self.act = nn.GELU()
        else:
            self.act = nn.Tanh()  # Default fallback

        # 5️⃣ Dense output
        self.fc = nn.Linear(rnn_hidden, 1)

    def forward(self, x):
        """
        x: (batch, time, features)
        """

        # ---- Conv1D ----
        x = x.transpose(1, 2)          # (batch, features, time)
        x = self.conv1d(x)
        x = x.transpose(1, 2)          # (batch, time, channels)

        # ---- LSTM ----
        x, _ = self.lstm(x)            # (batch, time, lstm_hidden)

        # ---- RNN ----
        x, _ = self.rnn(x)             # (batch, time, rnn_hidden)

        # ---- Take last timestep ----
        x = x[:, -1, :]                # (batch, rnn_hidden)

        # ---- Output ----
        out = self.fc(x)               # (batch, 1)
        return out.squeeze(-1)
    

class EarlyStopping:
    """
    Stops training if validation metric doesn't improve after a given patience.
    """
    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_weights = None

    def __call__(self, val_metric, model):
        # We assume lower val_metric (like RMSE or MAE) is better.
        score = -val_metric 

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            # print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        """Saves model weights when validation score improves."""
        # We use deepcopy so we don't just save a reference to the changing model
        self.best_weights = copy.deepcopy(model.state_dict())

    def restore_best_weights(self, model):
        """Loads the best weights back into the model."""
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)
            print("Restored best model weights!")

def evaluate(model, loader, device):
    model.eval()
    preds, targets = [], []

    with torch.no_grad():
        for X, y, _ in loader:
            X, y = X.to(device), y.to(device)
            p = model(X)
            # convert back
            p = torch.expm1(p)
            y = torch.expm1(y)
            preds.append(p.cpu().numpy())
            targets.append(y.cpu().numpy())
    
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)

    mse = np.mean((preds - targets) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(preds - targets))

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "preds": preds,
        "targets": targets,
    }

def evaluate_with_products(model, loader, device, product_features):
    model.eval()
    rows = []

    with torch.no_grad():
        for X, y, pid in loader:
            X, y = X.to(device), y.to(device)
            preds = model(X)
            # convert back
            preds = torch.expm1(preds)
            y = torch.expm1(y)
            for i in range(len(y)):
                rows.append({
                    "product_id_final": pid[i],
                    "y_true": y[i].item(),
                    "y_pred": preds[i].item(),
                })

    test_errors = pd.DataFrame(rows)
    test_errors["abs_error"] = np.abs(test_errors["y_pred"] - test_errors["y_true"])
    test_errors["sq_error"] = (test_errors["y_pred"] - test_errors["y_true"]) ** 2
    product_error_summary = (
    test_errors
    .groupby("product_id_final")
    .agg(
        n_windows=("abs_error", "count"),
        mae=("abs_error", "mean"),
        rmse=("sq_error", lambda x: np.sqrt(np.mean(x))),
        max_error=("abs_error", "max"),
        p90_error=("abs_error", lambda x: np.percentile(x, 90)),
    )
    .reset_index()
    )
    analysis_df = product_error_summary.merge(
    product_features,
    left_on="product_id_final",
    right_index=True,
    how="left"
    )
    threshold = analysis_df["rmse"].quantile(0.90)

    bad_products = analysis_df[analysis_df["rmse"] >= threshold]
    good_products = analysis_df[analysis_df["rmse"] < threshold]
    comparison = pd.DataFrame({
    "bad_products": bad_products.mean(numeric_only=True),
    "good_products": good_products.mean(numeric_only=True),
    "nº_bad": len(bad_products),
    "nº_good": len(good_products),
    })

    plt.scatter(
    analysis_df["std_qty"],
    analysis_df["rmse"],
    alpha=0.5
    )
    plt.xlabel("std")
    plt.ylabel("Product RMSE")
    plt.title("High Variability → Poor Predictability")
    plt.show()

    plt.scatter(
    analysis_df["mean_qty"],
    analysis_df["rmse"],
    alpha=0.5
    )
    plt.xlabel("mean_qty")
    plt.ylabel("Product RMSE")
    plt.title("High Variability → Poor Predictability")
    plt.show()
    print(comparison)
    print(bad_products.sort_values("rmse", ascending=False))

def naive_forecast(loader,n_target_weeks=3):
    preds, targets = [], []

    for X, y, _ in loader:
        preds.append(X[:, -1, 0].numpy()*n_target_weeks)  # last observed qty
        targets.append(y.numpy())

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)

    rmse = np.sqrt(np.mean((preds - targets) ** 2))
    mae = np.mean(np.abs(preds - targets))

    return rmse, mae

def flatten_windows(df, target_col="target"):
    X, y = [], []

    for _, row in df.iterrows():
        window = np.array(row["all_features"])  # (T, F)
        X.append(window.flatten())               # (T*F,)
        y.append(row[target_col])

    return np.array(X), np.array(y)

def extract_product_features_from_prod_df(prod_df):
    """
    prod_df: weekly dataframe for ONE product
             index = Fecha creación (weekly)
             columns: Cantidad en la cesta, Cliente
    """

    qty = prod_df["Cantidad en la cesta"].values
    cust_series = prod_df["Cliente"].tolist()
    T = len(qty)

    rolling_7 = [qty[max(0, i-6):i+1].sum() for i in range(T)]
    rolling_30 = [qty[max(0, i-29):i+1].sum() for i in range(T)]

    repeat_ratios = []
    num_customers = []
    days_without_purchase = []

    prev_cust = set()
    last_purchase_idx = -1

    for i in range(T):
        q = qty[i]
        c = cust_series[i] if isinstance(cust_series[i], set) else set()

        repeat = len(c & prev_cust) / len(c) if c else 0.0
        repeat_ratios.append(repeat)

        num_customers.append(len(c))

        if q > 0:
            last_purchase_idx = i
            days_without_purchase.append(0)
        else:
            days_without_purchase.append(
                i - last_purchase_idx if last_purchase_idx >= 0 else i + 1
            )

        prev_cust |= c

    qty = np.array(qty)

    return {
        "mean_qty": qty.mean(),
        "std_qty": qty.std(),
        "cv_qty": qty.std() / (qty.mean() + 1e-6),
        "zero_ratio": (qty == 0).mean(),
        "max_qty": qty.max(),
        "mean_rolling_7": np.mean(rolling_7),
        "mean_rolling_30": np.mean(rolling_30),
        "mean_customers": np.mean(num_customers),
        "mean_repeat_ratio": np.mean(repeat_ratios),
        "mean_days_without_purchase": np.mean(days_without_purchase),
        "max_days_without_purchase": np.max(days_without_purchase),
    }

def trainer_nn(model, train_loader, test_loader, device, optimizer, loss_fn, print_all=False, epochs=150, early_stop = (True,10), plot=False):
    train_mae_hist = []
    test_mae_hist = []
    train_rmse_hist = []
    test_rmse_hist = []
    
    #early stop
    stop=early_stop[0]
    if stop:
        patience=early_stop[1]

        # Initialize our referee
        early_stopping = EarlyStopping(patience=patience)
    
    for epoch in range(epochs):
        model.train()

        for X, y, _ in train_loader:
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            preds = model(X)
            loss = loss_fn(preds, y)
            loss.backward()
            optimizer.step()

        # Evaluate at the end of the epoch
        train_metrics = evaluate(model, train_loader, device)
        test_metrics  = evaluate(model, test_loader, device)
        
        train_mae_hist.append(train_metrics["mae"])
        test_mae_hist.append(test_metrics["mae"])
        train_rmse_hist.append(train_metrics["rmse"])
        test_rmse_hist.append(test_metrics["rmse"])
        
        if print_all:
            print(
                f"Epoch {epoch+1:02d} | "
                f"Train RMSE: {train_metrics['rmse']:.3f} | "
                f"Test RMSE: {test_metrics['rmse']:.3f} | "
                f"Train MAE: {train_metrics['mae']:.3f} | "
                f"Test MAE: {test_metrics['mae']:.3f}"
            )
        if stop:
            # ---- EARLY STOPPING CHECK ----
            # We pass the Test RMSE to the early stopping referee
            early_stopping(test_metrics["rmse"], model)
            
            if early_stopping.early_stop:
                print(f"Early stopping triggered at epoch {epoch+1}!")
                break # Exit the epoch loop!
    if stop:
        # ---- RESTORE BEST WEIGHTS ----
        # The loop finished (either naturally or via early stop), 
        # now we load the best weights back into the model.
        early_stopping.restore_best_weights(model)

        # Re-evaluate one last time with the best weights so our final metrics reflect the best model
        train_metrics = evaluate(model, train_loader, device)
        test_metrics = evaluate(model, test_loader, device)
    if plot:
        # Plotting code remains the same...
        plt.figure(figsize=(10, 4))
        plt.plot(test_metrics["targets"], label="Actual", alpha=0.7)
        plt.plot(test_metrics["preds"], label="Predicted", alpha=0.7)
        plt.legend()
        plt.title("Predicted vs Actual (Test Set - Best Weights)")
        plt.show()
        plt.close()

        plt.figure(figsize=(10, 4))
        plt.plot(train_rmse_hist, label="Train rmse")
        plt.plot(test_rmse_hist, label="Test rmse")
        # Mark where the best model was found
        if stop:
            best_epoch = len(train_rmse_hist) - early_stopping.counter - 1 if early_stopping.early_stop else len(train_rmse_hist) - 1
            plt.axvline(x=best_epoch, color='r', linestyle='--', label='Best Model')
        plt.xlabel("Epoch")
        plt.ylabel("RMSE")
        plt.title("RMSE over Epochs")
        plt.legend()
        plt.show()
        
    return test_metrics, model

def extract_inference_tensor(prod_df, window_size=30, row_qty_col="Cantidad en la cesta"):
    """
    Extracts the last 'window_size' weeks and perfectly recreates the 11 features 
    used in the training sliding_window function, formatting them as a PyTorch tensor.
    """
    # If a product has less than 30 weeks of history, pad it with zeros
    if len(prod_df) < window_size:
        pad_len = window_size - len(prod_df)
        pad_df = pd.DataFrame({row_qty_col: [0]*pad_len, "Cliente": [set()]*pad_len})
        prod_df = pd.concat([pad_df, prod_df])
        
    # Isolate the exact last 30 weeks
    recent_df = prod_df.iloc[-window_size:]
    window_qty = recent_df[row_qty_col].tolist()
    window_cust = recent_df["Cliente"].tolist()
    
    feature_window = []
    prev_cust = set()
    last_purchase_idx = -1
    
    # These must remain constant for the entire window to match your training logic
    mean_7 = np.mean(window_qty[-7:])
    std_7  = np.std(window_qty[-7:])
    mean_30 = np.mean(window_qty)
    std_30  = np.std(window_qty)
    
    cv_7  = std_7 / (mean_7 + 1e-6)
    cv_30 = std_30 / (mean_30 + 1e-6)
    
    for j in range(window_size):
        q = window_qty[j]
        c = window_cust[j] if isinstance(window_cust[j], set) else set()
        
        roll_7 = sum(window_qty[max(0, j - 6) : j + 1])
        roll_30 = sum(window_qty[max(0, j - 29) : j + 1])
        repeat = len(c & prev_cust) / len(c) if c else 0.0
        num_customers = len(c)
        
        if q > 0:
            days_without_purchase = 0
            last_purchase_idx = j
        else:
            days_without_purchase = j - last_purchase_idx if last_purchase_idx >= 0 else j + 1
            
        is_spike = int(q > mean_30 + 2 * std_30)
        
        # The 11 features in the EXACT order your network expects
        all_feats = [
            q,                      # 0
            roll_7,                 # 1
            roll_30,                # 2
            repeat,                 # 3
            num_customers,          # 4
            days_without_purchase,  # 5
            is_spike,               # 6
            cv_7,                   # 7
            cv_30,                  # 8
            std_7,                  # 9
            std_30                  # 10
        ]
        feature_window.append(all_feats)
        prev_cust |= c
        
    # Convert to PyTorch Tensor: shape becomes (1 batch, 30 time_steps, 11 features)
    tensor_input = torch.tensor([feature_window], dtype=torch.float32)
    return tensor_input
