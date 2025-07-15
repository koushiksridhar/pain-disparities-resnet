import torch
import torch.nn as nn
import torchvision.models as models
from dataset import get_loader
from model import ResNet18Regressor

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
import pandas as pd
from utils import setup_device
from config import Config
import argparse

def train_model(model, train_loader, val_loader, num_epochs=Config.NUM_EPOCHS, learning_rate=Config.LEARNING_RATE, device=None):
    # Ensure save directory exists
    Config.ensure_save_dir()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()  # Mean Squared Error for regression
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode=Config.SCHEDULER_MODE, factor=Config.SCHEDULER_FACTOR, 
        patience=Config.SCHEDULER_PATIENCE, verbose=True
    )
    
    best_val_loss = float('inf')

    history = {
        'train_loss': [], 'val_loss': [], 
        'train_rmse': [], 'val_rmse': [],
        'train_mae': [], 'val_mae': [],
        'train_r2': [], 'val_r2': [],
        'train_pearsonr': [], 'val_pearsonr': []
    }
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_outputs = []
        train_labels = []
        
        for batch_idx, batch in enumerate(train_loader):
            # Get data
            if torch.isnan(batch['img']).any():
                print(f'Nan in images, batch {batch_idx}')
                print(batch)
            if torch.isnan(batch['label']).any():
                print(f'Nan in labels, batch: {batch_idx}')
                print(batch)

            images = batch['img'].to(device)
            labels = batch['label'].float().to(device).view(-1, 1)  # Reshape to [batch_size, 1]
            masks = batch['img_mask'].to(device)

            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images, masks)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            train_outputs.extend(outputs.detach().cpu().numpy().flatten())
            train_labels.extend(labels.detach().cpu().numpy().flatten())
        
        train_loss = train_loss / len(train_loader.dataset)

        train_rmse = np.sqrt(mean_squared_error(train_labels, train_outputs))
        train_mae = mean_absolute_error(train_labels, train_outputs)
        train_pearson_r, _ = pearsonr(train_labels, train_outputs)
        train_r_squared = train_pearson_r ** 2
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_outputs = []
        val_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                if torch.isnan(batch['img']).any() or torch.isnan(batch['label']).any():
                    continue

                images = batch['img'].to(device)
                labels = batch['label'].float().to(device).view(-1, 1)
                masks = batch['img_mask'].to(device)
                
                outputs = model(images, masks)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                val_outputs.extend(outputs.cpu().numpy().flatten())
                val_labels.extend(labels.cpu().numpy().flatten())


        
        val_loss = val_loss / len(val_loader.dataset)
        val_rmse = np.sqrt(mean_squared_error(val_labels, val_outputs))
        val_mae = mean_absolute_error(val_labels, val_outputs)
        val_pearson_r, _ = pearsonr(val_labels, val_outputs)
        val_r_squared = val_pearson_r ** 2

        # Store metrics in history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_rmse'].append(train_rmse)
        history['val_rmse'].append(val_rmse)
        history['train_mae'].append(train_mae)
        history['val_mae'].append(val_mae)
        history['train_r2'].append(train_r_squared)
        history['val_r2'].append(val_r_squared)
        history['train_pearsonr'].append(train_pearson_r)
        history['val_pearsonr'].append(val_pearson_r)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_rmse': val_rmse,
                'val_mae': val_mae,
                'val_r2': val_r_squared,
                'val_pearsonr': val_pearson_r
            }, Config.get_save_path(Config.BEST_MODEL_NAME))    
            print('Saving best model')    

        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        print(f'Train RMSE: {train_rmse:.4f}, Val RMSE: {val_rmse:.4f}')
        print(f'Train MAE: {train_mae:.4f}, Val MAE: {val_mae:.4f}')
        print(f'Train R²: {train_r_squared:.4f}, Val R²: {val_r_squared:.4f}')
        print(f'Train Pearson r: {train_pearson_r:.4f}, Val Pearson r: {val_pearson_r:.4f}')
        print('-' * 60)

    # Load best model
    checkpoint = torch.load(Config.get_save_path(Config.BEST_MODEL_NAME))
    model.load_state_dict(checkpoint['model_state_dict'])    
    
    # Save training history to CSV
    history_df = pd.DataFrame(history)
    history_df.to_csv(Config.get_save_path(Config.TRAINING_HISTORY_NAME), index=False)
    
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    return model

def predict(model, test_loader, device=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Evaluating on {device}')

    model = model.to(device)
    model.eval()
    
    predictions = []
    names = []
    actual = []
    
    with torch.no_grad():
        for batch in test_loader:
            if torch.isnan(batch['img']).any():
                continue
                
            images = batch['img'].to(device)
            masks = batch['img_mask'].to(device)
            batch_names = batch['name']
            
            # If labels are available in test set
            if 'label' in batch and not torch.isnan(batch['label']).any():
                labels = batch['label'].float().cpu().numpy().flatten()
                actual.extend(labels)
            
            outputs = model(images, masks)
            
            predictions.extend(outputs.cpu().numpy().flatten())
            names.extend(batch_names)
    
    results = pd.DataFrame({
        'name': names,
        'predicted_WOMAC': predictions
    })
    
    # If we have ground truth, calculate and report metrics
    if len(actual) > 0:
        rmse = np.sqrt(mean_squared_error(actual, predictions))
        mae = mean_absolute_error(actual, predictions)
        pearson_r, p_value = pearsonr(actual, predictions)
        r_squared = pearson_r ** 2
        
        print(f"Test Set Metrics:")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R²: {r_squared:.4f}")
        print(f"Pearson r: {pearson_r:.4f} (p-value: {p_value:.4f})")
        
        # Add actual values to results
        results['actual_WOMAC'] = actual
        results['error'] = np.abs(results['actual_WOMAC'] - results['predicted_WOMAC'])
    
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train ResNet18 model for WOMAC prediction')
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='gpu device ids')
    parser.add_argument('--train', type=str, default='true', help='whether to train the model (true/false)')

    args = parser.parse_args()
    device_ = setup_device(args)

    # Handle train argument
    if args.train.lower() == 'true':
        print("train")
        train_loader = get_loader(phase='train', shuffle=True)
        val_loader = get_loader(phase='val', shuffle=False)
        test_loader = get_loader(phase='test', shuffle=False)
    elif args.train.lower() == 'false':
        print("false")
        test_loader = get_loader(phase='test', shuffle=False)
    else:
        print(f"Invalid train argument: {args.train}. Expected 'true' or 'false'")

    print("Loaded data")


    print("Initializing model...")
    # Initialize the model
    model = ResNet18Regressor(pretrained=Config.PRETRAINED)

    if args.train.lower() == 'true':
        print("Starting training...")
        # Train the model
        trained_model = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=Config.NUM_EPOCHS,
            learning_rate=Config.LEARNING_RATE,
            device=device_
        )
    else:
        print("Loading best model...")
        model.load_state_dict(torch.load(Config.get_save_path(Config.BEST_MODEL_NAME))['model_state_dict'])
        trained_model = model

    print("Evaluating on test set...")
    # Evaluate on test set with the best trained model
    results = predict(trained_model, test_loader, device=device_)
    results.to_csv(Config.get_save_path(Config.RESULTS_NAME), index=False)
    
    print(f"Training complete! Results saved to {Config.get_save_path(Config.RESULTS_NAME)}")