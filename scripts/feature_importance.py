import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import argparse
import logging
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
from sklearn.inspection import permutation_importance
from sklearn.base import BaseEstimator, ClassifierMixin

# Custom local imports
import utils
from PET import PET
from omnifold import Classifier

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Compute feature importance using permutation importance for trained models.")
    parser.add_argument("--dataset", type=str, default="jetclass", help="Dataset to use")
    parser.add_argument("--folder", type=str, default="tau/", help="Folder containing input files")
    parser.add_argument("--mode", type=str, default="classifier", help="Loss type used to train the model")
    parser.add_argument("--batch", type=int, default=100, help="Batch size for evaluation")
    parser.add_argument("--nsamples", type=int, default=1000, help="Number of samples for permutation importance")
    parser.add_argument("--n_repeats", type=int, default=10, help="Number of permutation repeats")
    parser.add_argument("--fine_tune", action='store_true', default=False, help='Fine tuned model')
    parser.add_argument("--local", action='store_true', default=False, help='Use local embedding')
    parser.add_argument("--num_layers", type=int, default=8, help="Number of transformer layers")
    parser.add_argument("--drop_probability", type=float, default=0.0, help="Drop probability")
    parser.add_argument("--simple", action='store_true', default=False, help='Use simplified head model')
    parser.add_argument("--talking_head", action='store_true', default=False, help='Use talking head attention')
    parser.add_argument("--layer_scale", action='store_true', default=False, help='Use layer scale')
    parser.add_argument("--nid", type=int, default=0, help="Training ID for multiple trainings")
    parser.add_argument("--max_display", type=int, default=20, help="Max features to display in plots")
    return parser.parse_args()


def get_data_loader(flags):
    """Load the appropriate test dataset"""
    if flags.dataset == 'top':
        test = utils.TopDataLoader(os.path.join(flags.folder,'TOP', 'test_ttbar.h5'),
                                   flags.batch, 0, 1)
        folder_name = 'TOP'
    elif flags.dataset == 'tau':
        test = utils.TauDataLoader(os.path.join(flags.folder,'TAU', 'test_tau.h5'),
                                   flags.batch, 0, 1)
        folder_name = 'TAU'
    elif flags.dataset == 'qg':
        test = utils.QGDataLoader(os.path.join(flags.folder,'QG', 'test_qg.h5'),
                                  flags.batch, 0, 1)
        folder_name = 'QG'
    elif flags.dataset == 'cms':
        test = utils.CMSQGDataLoader(os.path.join(flags.folder,'CMSQG', 'test_qgcms_pid.h5'),
                                     flags.batch, 0, 1)
        folder_name = 'CMSQG'
    elif flags.dataset == 'h1':
        test = utils.H1DataLoader(os.path.join(flags.folder,'H1', 'test.h5'),
                                  flags.batch, 0, 1)
        folder_name = 'H1'
    elif flags.dataset == 'atlas':
        test = utils.AtlasDataLoader(os.path.join(flags.folder,'ATLASTOP', 'test_atlas.h5'),
                                     flags.batch, 0, 1)
        folder_name = 'ATLASTOP'
    elif flags.dataset == 'jetclass':
        test = utils.JetClassDataLoader(os.path.join(flags.folder,'JetClass','test'),
                                        flags.batch, 0, 1)
        folder_name = 'JetClass/test'
    else:
        raise ValueError(f"Unknown dataset: {flags.dataset}")
    
    return test, folder_name


def get_model_function(dataset):
    """Determine which model class to use"""
    if 'atlas' in dataset:
        return Classifier, 'sigmoid'
    else:
        return PET, 'softmax'


def load_model(flags, test):
    """Load the trained model"""
    model_function, activation = get_model_function(flags.dataset)
    
    model = model_function(
        num_feat=test.num_feat,
        num_jet=test.num_jet,
        num_classes=test.num_classes,
        local=flags.local,
        num_layers=flags.num_layers,
        drop_probability=flags.drop_probability,
        simple=flags.simple,
        layer_scale=flags.layer_scale,
        talking_head=flags.talking_head,
        mode=flags.mode,
        class_activation=activation
    )
    
    add_string = '_{}'.format(flags.nid) if flags.nid > 0 else ''
    model_path = os.path.join(
        flags.folder, 'checkpoints',
        utils.get_model_name(flags, fine_tune=flags.fine_tune, add_string=add_string)
    )
    
    logger.info(f"Loading model from: {model_path}")
    model.load_weights(model_path)
    
    return model


class KerasModelWrapper(BaseEstimator, ClassifierMixin):
    """Wrapper to make Keras model compatible with sklearn permutation_importance"""
    
    def __init__(self, model, X_template, n_particles, n_part_features, n_jet_features):
        self.model = model
        self.X_template = X_template  # Template for reconstructing input dict
        self.n_particles = n_particles
        self.n_part_features = n_part_features
        self.n_jet_features = n_jet_features
    
    def fit(self, X, y):
        """Dummy fit method - model is already trained"""
        return self
    
    def _reconstruct_inputs(self, X_flat):
        """Reconstruct model inputs from flattened feature array"""
        n_samples = X_flat.shape[0]
        
        # Split flattened array back into particle features and jet features
        n_part_total = self.n_particles * self.n_part_features
        
        part_features_flat = X_flat[:, :n_part_total]
        jet_features = X_flat[:, n_part_total:]
        
        # Reshape particle features back to [n_samples, n_particles, n_features]
        features_3d = part_features_flat.reshape(n_samples, self.n_particles, self.n_part_features)
        
        # Construct full input dictionary
        inputs = {
            'input_features': features_3d,
            'input_points': features_3d[:, :, :2],
            'input_mask': self.X_template['input_mask'][:n_samples],
            'input_jet': jet_features,
            'input_time': np.zeros((n_samples, 1), dtype=np.float32)
        }
        return inputs
        
    def predict(self, X_flat):
        """Predict using flattened features (particle + jet)"""
        inputs = self._reconstruct_inputs(X_flat)
        preds = self.model(inputs, training=False)
        if isinstance(preds, (list, tuple)):
            preds = preds[0]
        return np.argmax(preds.numpy(), axis=-1)
    
    def predict_proba(self, X_flat):
        """Predict probabilities"""
        inputs = self._reconstruct_inputs(X_flat)
        preds = self.model(inputs, training=False)
        if isinstance(preds, (list, tuple)):
            preds = preds[0]
        return preds.numpy()
    
    def score(self, X_flat, y):
        """Compute accuracy score"""
        y_pred = self.predict(X_flat)
        if len(y.shape) > 1:
            y = np.argmax(y, axis=-1)
        return np.mean(y_pred == y)


def prepare_data(test_loader, nsamples):
    """Prepare test data for permutation importance"""
    logger.info(f"Loading {nsamples} samples...")
    
    # make_eval_data returns: X (tf.data.Dataset), y (numpy), event_id, weight
    X_dataset, y_full, event_id, weight = test_loader.make_eval_data()
    
    # Convert tf.data.Dataset to numpy arrays
    X_dict = {}
    for key in X_dataset.element_spec.keys():
        X_dict[key] = []
    
    samples_collected = 0
    for batch_X in X_dataset:
        batch_size = batch_X['input_features'].shape[0]
        for key in batch_X.keys():
            X_dict[key].append(batch_X[key].numpy())
        samples_collected += batch_size
        if samples_collected >= nsamples:
            break
    
    # Concatenate all batches
    for key in X_dict.keys():
        X_dict[key] = np.concatenate(X_dict[key], axis=0)
    
    # Trim to exact number of samples
    X_dict = {key: val[:nsamples] for key, val in X_dict.items()}
    
    # y_full is already a numpy array - just slice it
    y_array = y_full[:nsamples]
    
    logger.info(f"Data shape: {X_dict['input_features'].shape}, y shape: {y_array.shape}")
    logger.info(f"y unique values: {np.unique(y_array if len(y_array.shape)==1 else np.argmax(y_array, axis=1), return_counts=True)}")
    
    return X_dict, y_array


def compute_permutation_importance(model, X_dict, y, flags, part_names=None, jet_names=None):
    """Compute permutation importance using sklearn for both particle and jet features"""
    logger.info("Computing permutation importance...")
    
    n_samples, n_particles, n_part_features = X_dict['input_features'].shape
    n_jet_features = X_dict['input_jet'].shape[1]
    
    # Flatten particle features: [n_samples, n_particles * n_part_features]
    X_part_flat = X_dict['input_features'].reshape(n_samples, -1)
    
    # Concatenate particle and jet features: [n_samples, n_particles * n_part_features + n_jet_features]
    X_flat = np.concatenate([X_part_flat, X_dict['input_jet']], axis=1)
    
    # Convert y to class indices if one-hot encoded
    if len(y.shape) > 1 and y.shape[1] > 1:
        y = np.argmax(y, axis=-1)
    y = np.asarray(y).ravel()
    
    logger.info(f"X_flat shape: {X_flat.shape}, y shape: {y.shape}")
    logger.info(f"  - Particle features: {n_particles} particles x {n_part_features} features = {n_particles * n_part_features} columns")
    logger.info(f"  - Jet features: {n_jet_features} columns")
    
    # Create sklearn-compatible wrapper
    wrapper = KerasModelWrapper(model, X_dict, n_particles, n_part_features, n_jet_features)
    
    # Check baseline accuracy first
    baseline_acc = wrapper.score(X_flat, y)
    logger.info(f"Baseline model accuracy on this data: {baseline_acc:.4f}")
    
    # Compute permutation importance
    logger.info(f"Running permutation importance with {flags.n_repeats} repeats on {X_flat.shape[1]} feature columns...")
    
    result = permutation_importance(
        wrapper, X_flat, y,
        n_repeats=flags.n_repeats,
        random_state=42,
        n_jobs=1,
        scoring='accuracy'
    )
    
    # Split importance results back into particle and jet features
    n_part_total = n_particles * n_part_features
    
    # Particle feature importance
    part_importance_per_column = result.importances_mean[:n_part_total]
    part_std_per_column = result.importances_std[:n_part_total]
    
    # Reshape to [n_particles, n_part_features] and aggregate across particles
    part_importance_reshaped = part_importance_per_column.reshape(n_particles, n_part_features)
    part_std_reshaped = part_std_per_column.reshape(n_particles, n_part_features)
    
    # Sum across particles to get per-feature importance for particle features
    part_feature_importance = np.sum(part_importance_reshaped, axis=0)
    part_feature_std = np.sqrt(np.sum(part_std_reshaped**2, axis=0))
    
    # Jet feature importance (already per-feature)
    jet_feature_importance = result.importances_mean[n_part_total:]
    jet_feature_std = result.importances_std[n_part_total:]
    
    # Combine all feature importances
    all_importance = np.concatenate([part_feature_importance, jet_feature_importance])
    all_std = np.concatenate([part_feature_std, jet_feature_std])
    
    # Create combined feature names
    if part_names is None:
        part_names = [f'Track_Feature_{i}' for i in range(n_part_features)]
    if jet_names is None:
        jet_names = [f'Jet_Feature_{i}' for i in range(n_jet_features)]
    
    # Add prefix to distinguish feature types
    part_names_prefixed = [f'Track: {name}' for name in part_names]
    jet_names_prefixed = [f'Jet: {name}' for name in jet_names]
    all_feature_names = part_names_prefixed + jet_names_prefixed
    
    # Sort by importance
    sorted_indices = np.argsort(all_importance)[::-1]
    
    importance_dict = {
        'importance': all_importance,
        'importance_std': all_std,
        'sorted_indices': sorted_indices,
        'feature_names': all_feature_names,
        'raw_importances': result.importances,
        # Keep separate for detailed analysis
        'particle_importance': part_feature_importance,
        'particle_importance_std': part_feature_std,
        'particle_importance_per_particle': part_importance_reshaped,
        'particle_names': part_names,
        'jet_importance': jet_feature_importance,
        'jet_importance_std': jet_feature_std,
        'jet_names': jet_names,
        'n_particles': n_particles,
        'n_part_features': n_part_features,
        'n_jet_features': n_jet_features,
    }
    
    return importance_dict


def plot_feature_importance(importance_dict, output_path, max_display=20):
    """Create bar plot of feature importance"""
    importance = importance_dict['importance']
    importance_std = importance_dict.get('importance_std', None)
    sorted_indices = importance_dict['sorted_indices']
    feature_names = importance_dict['feature_names']
    
    # Get top features
    top_indices = sorted_indices[:max_display]
    top_importance = importance[top_indices]
    top_std = importance_std[top_indices] if importance_std is not None else None
    
    if feature_names is not None:
        top_names = [feature_names[i] for i in top_indices]
    else:
        top_names = [f'Feature {i}' for i in top_indices]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    y_pos = range(len(top_indices))
    
    if top_std is not None:
        ax.barh(y_pos, top_importance, xerr=top_std, capsize=3, color='steelblue', alpha=0.8)
    else:
        ax.barh(y_pos, top_importance, color='steelblue', alpha=0.8)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_names)
    ax.set_xlabel('Permutation Importance (Accuracy Drop)')
    ax.set_title(f'Top {max_display} Feature Importance')
    ax.invert_yaxis()
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved feature importance plot to: {output_path}")
    plt.close()


def plot_importance_per_particle(importance_dict, output_path, max_particles=10):
    """Create heatmap of feature importance per particle (for particle features only)"""
    importance_per_particle = importance_dict.get('particle_importance_per_particle', None)
    if importance_per_particle is None:
        logger.warning("No per-particle importance data available for heatmap")
        return
    
    n_particles, n_features = importance_per_particle.shape
    feature_names = importance_dict.get('particle_names', None)
    
    # Limit particles for visualization
    if n_particles > max_particles:
        # Sum importance across features for each particle
        particle_total_importance = np.sum(importance_per_particle, axis=1)
        top_particle_indices = np.argsort(particle_total_importance)[-max_particles:]
        importance_subset = importance_per_particle[top_particle_indices]
        particle_labels = [f'Track {i}' for i in top_particle_indices]
    else:
        importance_subset = importance_per_particle
        particle_labels = [f'Track {i}' for i in range(n_particles)]
    
    if feature_names is None:
        feature_names = [f'Feat {i}' for i in range(n_features)]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(importance_subset, cmap='YlOrRd', aspect='auto')
    
    ax.set_xticks(range(n_features))
    ax.set_xticklabels(feature_names, rotation=45, ha='right')
    ax.set_yticks(range(len(particle_labels)))
    ax.set_yticklabels(particle_labels)
    ax.set_xlabel('Features')
    ax.set_ylabel('Tracks')
    ax.set_title('Track Feature Importance by Track Index')
    
    plt.colorbar(im, ax=ax, label='Importance')
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved importance heatmap to: {output_path}")
    plt.close()


def plot_separate_importance(importance_dict, output_dir, max_display=15):
    """Create separate bar plots for track and jet features"""
    
    # Plot track features
    part_importance = importance_dict['particle_importance']
    part_std = importance_dict['particle_importance_std']
    part_names = importance_dict['particle_names']
    
    sorted_part_idx = np.argsort(part_importance)[::-1][:max_display]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = range(len(sorted_part_idx))
    ax.barh(y_pos, part_importance[sorted_part_idx], 
            xerr=part_std[sorted_part_idx], capsize=3, color='steelblue', alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([part_names[i] for i in sorted_part_idx])
    ax.set_xlabel('Permutation Importance (Accuracy Drop)')
    ax.set_title('Track Feature Importance (aggregated across all tracks)')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'track_feature_importance.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot jet features
    jet_importance = importance_dict['jet_importance']
    jet_std = importance_dict['jet_importance_std']
    jet_names = importance_dict['jet_names']
    
    sorted_jet_idx = np.argsort(jet_importance)[::-1][:max_display]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = range(len(sorted_jet_idx))
    ax.barh(y_pos, jet_importance[sorted_jet_idx],
            xerr=jet_std[sorted_jet_idx], capsize=3, color='darkorange', alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([jet_names[i] for i in sorted_jet_idx])
    ax.set_xlabel('Permutation Importance (Accuracy Drop)')
    ax.set_title('Jet/DiTau Feature Importance')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'jet_feature_importance.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved separate track and jet importance plots to: {output_dir}")


def save_results(importance_dict, output_dir, flags):
    """Save importance values and rankings"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save importance dictionary
    importance_file = os.path.join(output_dir, 'feature_importance.pkl')
    with open(importance_file, 'wb') as f:
        pickle.dump(importance_dict, f)
    logger.info(f"Saved feature importance to: {importance_file}")
    
    # Save raw importance arrays
    np.save(os.path.join(output_dir, 'importance_all.npy'), importance_dict['importance'])
    np.save(os.path.join(output_dir, 'importance_std_all.npy'), importance_dict['importance_std'])
    np.save(os.path.join(output_dir, 'particle_importance.npy'), importance_dict['particle_importance'])
    np.save(os.path.join(output_dir, 'jet_importance.npy'), importance_dict['jet_importance'])
    if 'particle_importance_per_particle' in importance_dict:
        np.save(os.path.join(output_dir, 'importance_per_track.npy'), 
                importance_dict['particle_importance_per_particle'])
    
    # Save importance rankings as text
    rankings_file = os.path.join(output_dir, 'feature_rankings.txt')
    with open(rankings_file, 'w') as f:
        f.write("Feature Importance Rankings\n")
        f.write("=" * 50 + "\n\n")
        
        importance = importance_dict['importance']
        sorted_indices = importance_dict['sorted_indices']
        feature_names = importance_dict['feature_names']
        
        for rank, idx in enumerate(sorted_indices, 1):
            name = feature_names[idx] if feature_names else f'Feature {idx}'
            f.write(f"{rank:3d}. {name:30s} : {importance[idx]:.6f}\n")
    
    logger.info(f"Saved feature rankings to: {rankings_file}")


def main():
    utils.setup_gpus()
    flags = parse_arguments()
    
    # Load data
    test_loader, folder_name = get_data_loader(flags)
    
    # Load model
    model = load_model(flags, test_loader)
    
    # Prepare data
    X_dict, y = prepare_data(test_loader, flags.nsamples)
    
    # Get feature names if available
    part_names = getattr(test_loader, 'part_names', None)
    if part_names is None:
        part_names = [f'Track_Feature_{i}' for i in range(test_loader.num_feat)]
    
    jet_names = getattr(test_loader, 'jet_names', None)
    if jet_names is None:
        jet_names = [f'Jet_Feature_{i}' for i in range(test_loader.num_jet)]
    
    logger.info(f"Track features ({len(part_names)}): {part_names}")
    logger.info(f"Jet features ({len(jet_names)}): {jet_names}")
    
    # Compute permutation importance
    importance_dict = compute_permutation_importance(
        model, X_dict, y, flags, part_names, jet_names
    )
    
    # Create output directory
    model_name = utils.get_model_name(
        flags, flags.fine_tune,
        add_string='_{}'.format(flags.nid) if flags.nid > 0 else ''
    ).replace('.weights.h5', '')
    output_dir = os.path.join(flags.folder, folder_name, 'feature_importance', model_name)
    
    # Save results
    save_results(importance_dict, output_dir, flags)
    
    # Create plots
    # Combined plot
    plot_path = os.path.join(output_dir, 'feature_importance_all.png')
    plot_feature_importance(importance_dict, plot_path, max_display=flags.max_display)
    
    # Separate plots for track and jet features
    plot_separate_importance(importance_dict, output_dir, max_display=flags.max_display)
    
    # Heatmap of track features per track
    heatmap_path = os.path.join(output_dir, 'importance_per_track.png')
    plot_importance_per_particle(importance_dict, heatmap_path, max_particles=10)
    
    # Print top features (combined)
    feature_names = importance_dict['feature_names']
    logger.info("\n" + "="*60)
    logger.info("Top 15 Most Important Features (All):")
    logger.info("="*60)
    for rank, idx in enumerate(importance_dict['sorted_indices'][:15], 1):
        name = feature_names[idx]
        imp = importance_dict['importance'][idx]
        std = importance_dict['importance_std'][idx]
        logger.info(f"{rank:2d}. {name:40s} : {imp:.6f} +/- {std:.6f}")
    
    # Print track features separately
    logger.info("\n" + "="*60)
    logger.info("Track Feature Importance:")
    logger.info("="*60)
    part_sorted = np.argsort(importance_dict['particle_importance'])[::-1]
    for rank, idx in enumerate(part_sorted, 1):
        name = part_names[idx]
        imp = importance_dict['particle_importance'][idx]
        std = importance_dict['particle_importance_std'][idx]
        logger.info(f"{rank:2d}. {name:30s} : {imp:.6f} +/- {std:.6f}")
    
    # Print jet features separately
    logger.info("\n" + "="*60)
    logger.info("Jet/DiTau Feature Importance:")
    logger.info("="*60)
    jet_sorted = np.argsort(importance_dict['jet_importance'])[::-1]
    for rank, idx in enumerate(jet_sorted, 1):
        name = jet_names[idx]
        imp = importance_dict['jet_importance'][idx]
        std = importance_dict['jet_importance_std'][idx]
        logger.info(f"{rank:2d}. {name:30s} : {imp:.6f} +/- {std:.6f}")
    
    logger.info("="*60)
    logger.info("Feature importance analysis complete!")


if __name__ == "__main__":
    main()
