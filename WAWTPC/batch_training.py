from tqdm import tqdm

def train_model(model, train_dataset, val_dataset, n_epoch, patience, log_dir, evaluate_every_batches = 100):
    best_weights = None
    best_loss = float('inf')
    wait = 0

    for epoch in range(n_epoch):
        print(f"Starting epoch {epoch+1}/{n_epoch}")
        for i, data in enumerate(tqdm(train_dataset)):
            images, target = data
            loss, mse = model.train_on_batch(images, target)
            if i % evaluate_every_batches == 0:
                val_loss = evaluate_model(model, val_dataset.take(5))
                if val_loss < best_loss:
                    print(f"Training loss {loss} Validation loss improved to {val_loss:.4f}")
                    best_loss = val_loss
                    best_weights = model.get_weights()
                    model.save(log_dir / f'model-{epoch}-{i}.h5')
                    wait = 0
                else:
                    wait += 1
                    print(f"\tTraining loss {loss} Validation loss {val_loss:.4f} not improved from best loss {best_loss:.4f}")
                    if wait >= patience:
                        print(f"No improvement in {patience} steps, restoring best weights and stopping training")
                        model.set_weights(best_weights)
                        return
        print(f"Finished epoch {epoch+1}/{n_epoch}")

def evaluate_model(model, val_dataset):
    total_loss = 0
    total_batches = 0
    for val_images, val_target in val_dataset:
        val_loss, _ = model.evaluate(val_images, val_target, verbose=0)
        total_loss += val_loss
        total_batches += 1
    return total_loss / total_batches