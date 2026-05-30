"""
Comprehensive fix script for quack-track (patch_v2).
Run in Colab:  python /content/quack-track/scripts/patch_v2.py

Changes:
1. Anchor centering: _anchors(), _build_anchors(), compute_targets() (fixes top-left shift)
2. Debug prints: positive anchor count, max IoU, warning when zero
3. src/data/split.py: deterministic sequence-level train/val/test split
4. dataset.py: accept seq_names filter param
5. trainer.py: validation loop, early stopping, best-val checkpoint
6. scripts/train.py: fix arg name (--checkpoint-dir)
7. scripts/test.py: default to best_model.pth + --debug flag
8. configs/default.yaml: epochs=80, patience=15
"""

import os

BASE = "/content/quack-track"


def fix_train_arg():
    path = os.path.join(BASE, "scripts", "train.py")
    with open(path) as f:
        c = f.read()
    c = c.replace(
        "parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')",
        "parser.add_argument('--checkpoint-dir', type=str, default='checkpoints')"
    )
    with open(path, 'w') as f:
        f.write(c)
    print("[OK] train.py: --checkpoint_dir -> --checkpoint-dir")


def add_debug_to_compute_targets():
    path = os.path.join(BASE, "src", "trainer.py")
    with open(path) as f:
        c = f.read()

    # Check if adaptive code is already present (from repo commit)
    if 'pos_threshold = max(0.1, max_iou * 0.5)' in c:
        print("[OK] compute_targets: adaptive threshold + debug already present")
        return

    old = """    for i in range(response_sz):
        for j in range(response_sz):
            for k in range(5):
                anc_box = anc[i, j, k]
                anc_w = float(anc_box[2])
                anc_h = float(anc_box[3])
                if anc_w <= 0 or anc_h <= 0:
                    continue
                iou = _bbox_iou(anc_box, [gx, gy, gw, gh])
                idx = (i * response_sz + j) * 5 + k
                if iou > 0.5:
                    reg_target[0, idx] = float((gx - anc_box[0]) / anc_w)
                    reg_target[1, idx] = float((gy - anc_box[1]) / anc_h)
                    reg_target[2, idx] = math.log(gw / anc_w)
                    reg_target[3, idx] = math.log(gh / anc_h)
                    reg_mask[idx] = 1.0

    return cls_target, reg_target, reg_mask"""

    new = """    max_iou = -1.0
    for i in range(response_sz):
        for j in range(response_sz):
            for k in range(5):
                anc_box = anc[i, j, k]
                anc_w = float(anc_box[2])
                anc_h = float(anc_box[3])
                if anc_w > 0 and anc_h > 0:
                    iou = _bbox_iou(anc_box, [gx, gy, gw, gh])
                    max_iou = max(max_iou, iou)

    pos_threshold = max(0.1, max_iou * 0.5) if max_iou > 0 else 1.0

    pos_count = 0
    for i in range(response_sz):
        for j in range(response_sz):
            for k in range(5):
                anc_box = anc[i, j, k]
                anc_w = float(anc_box[2])
                anc_h = float(anc_box[3])
                if anc_w <= 0 or anc_h <= 0:
                    continue
                iou = _bbox_iou(anc_box, [gx, gy, gw, gh])
                idx = (i * response_sz + j) * 5 + k
                if iou > 0 and iou >= pos_threshold:
                    pos_count += 1
                    reg_target[0, idx] = float((gx - anc_box[0]) / anc_w)
                    reg_target[1, idx] = float((gy - anc_box[1]) / anc_h)
                    reg_target[2, idx] = math.log(gw / anc_w)
                    reg_target[3, idx] = math.log(gh / anc_h)
                    reg_mask[idx] = 1.0

    if pos_count == 0:
        logger.warning(
            'compute_targets: ZERO positive anchors (max_iou=%.3f, gw=%.1f gh=%.1f '
            'gx=%.1f gy=%.1f)', max_iou, gw, gh, gx, gy
        )

    return cls_target, reg_target, reg_mask"""

    assert old in c, "compute_targets loop not found"
    c = c.replace(old, new)
    with open(path, 'w') as f:
        f.write(c)
    print("[OK] compute_targets: added adaptive threshold + debug logging")


def create_split_module():
    path = os.path.join(BASE, "src", "data", "split.py")
    os.makedirs(os.path.join(BASE, "src", "data"), exist_ok=True)
    with open(path, 'w') as f:
        f.write('''\
import os
import random
import logging

from .dataset import _list_sequences, UAV20L_SEQS

logger = logging.getLogger('quack-track')


def create_uav123_split(data_root, val_ratio=0.1, test_ratio=0.1, seed=42,
                        exclude_uav20l=True):
    """Deterministic sequence-level train/val/test split. Returns (train, val, test)
    as sets of sequence names."""
    seq_entries = _list_sequences(data_root)
    seq_names = sorted(set(n for n, _ in seq_entries))

    if exclude_uav20l:
        seq_names = [n for n in seq_names if n not in UAV20L_SEQS]

    rng = random.Random(seed)
    rng.shuffle(seq_names)

    n = len(seq_names)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)

    test_seqs = set(seq_names[:n_test])
    val_seqs = set(seq_names[n_test:n_test + n_val])
    train_seqs = set(seq_names[n_test + n_val:])

    logger.info(
        'Dataset split: %d train, %d val, %d test (seed=%d, exclude_uav20l=%s)',
        len(train_seqs), len(val_seqs), len(test_seqs), seed, exclude_uav20l,
    )

    return train_seqs, val_seqs, test_seqs
''')
    print("[OK] Created src/data/split.py")


def fix_dataset_seq_names():
    path = os.path.join(BASE, "src", "data", "dataset.py")
    with open(path) as f:
        c = f.read()

    c = c.replace(
        "exclude_seqs=None):",
        "exclude_seqs=None, seq_names=None):"
    )

    c = c.replace(
        "seq_entries = _list_sequences(data_root)\n        split_path = os.path.join(data_root, f'{split}.json')",
        "seq_entries = _list_sequences(data_root)\n\n        if seq_names is not None:\n            seq_entries = [(n, d) for n, d in seq_entries if n in seq_names]\n\n        split_path = os.path.join(data_root, f'{split}.json')"
    )

    c = c.replace(
        "if os.path.exists(split_path):",
        "if os.path.exists(split_path) and seq_names is None:"
    )

    if "import logging" not in c:
        c = c.replace(
            "from torch.utils.data import Dataset",
            "import logging\nfrom torch.utils.data import Dataset"
        )

    with open(path, 'w') as f:
        f.write(c)
    print("[OK] dataset.py: added seq_names param")


def rewrite_trainer():
    path = os.path.join(BASE, "src", "trainer.py")
    with open(path) as f:
        c = f.read()

    old_init = """class Trainer:
    def __init__(self, model, cfg, device='cuda'):
        self.model = model
        self.cfg = cfg
        self.device = device

        self.criterion = SiamRPNLoss(
            cls_weight=cfg['loss']['cls_weight'],
            reg_weight=cfg['loss']['reg_weight']
        )

        self.optimizer = torch.optim.SGD(
            model.parameters(),
            lr=cfg['train']['lr'],
            momentum=cfg['train']['momentum'],
            weight_decay=cfg['train']['weight_decay']
        )

        total_epochs = cfg['train']['epochs']
        warmup_epochs = cfg['train']['warmup_epochs']
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=total_epochs - warmup_epochs
        )
        self.warmup_epochs = warmup_epochs

        self.start_epoch = 0
        self.best_loss = float('inf')"""

    new_init = """class Trainer:
    def __init__(self, model, cfg, device='cuda'):
        self.model = model
        self.cfg = cfg
        self.device = device

        self.criterion = SiamRPNLoss(
            cls_weight=cfg['loss']['cls_weight'],
            reg_weight=cfg['loss']['reg_weight']
        )

        self.optimizer = torch.optim.SGD(
            model.parameters(),
            lr=cfg['train']['lr'],
            momentum=cfg['train']['momentum'],
            weight_decay=cfg['train']['weight_decay']
        )

        total_epochs = cfg['train']['epochs']
        warmup_epochs = cfg['train']['warmup_epochs']
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=total_epochs - warmup_epochs
        )
        self.warmup_epochs = warmup_epochs

        self.start_epoch = 0
        self.best_val_loss = float('inf')
        self.early_stop_counter = 0
        self.patience = cfg['train'].get('patience', 15)
        self.val_interval = cfg['train'].get('val_interval', 1)
        self.val_loader = None"""

    assert old_init in c, "Trainer.__init__ not found"
    c = c.replace(old_init, new_init)

    old_build = """    def _build_dataloaders(self):
        cfg = self.cfg
        exclude = None
        if cfg['data'].get('uav20l_exclude', True):
            exclude = UAV20L_SEQS
        tracking = UAVTrackingDataset(
            data_root=cfg['data']['uav123_root'],
            pairs_per_seq=cfg['data']['pairs_per_seq'],
            exclude_seqs=exclude
        )
        if cfg['data'].get('coco_root'):
            coco = COCONegativeDataset(
                coco_root=cfg['data']['coco_root'],
                num_samples=cfg['data'].get('coco_samples', 50000)
            )
            dataset = ConcatDataset([tracking, coco])
        else:
            dataset = tracking

        return DataLoader(
            dataset,
            batch_size=cfg['train']['batch_size'],
            shuffle=True,
            num_workers=cfg['train']['num_workers'],
            collate_fn=collate_tracking,
            drop_last=True
        )"""

    new_build = """    def _build_dataloaders(self):
        cfg = self.cfg

        from .data.split import create_uav123_split
        train_seqs, val_seqs, _ = create_uav123_split(
            data_root=cfg['data']['uav123_root'],
            exclude_uav20l=cfg['data'].get('uav20l_exclude', True),
        )

        train_dataset = UAVTrackingDataset(
            data_root=cfg['data']['uav123_root'],
            pairs_per_seq=cfg['data']['pairs_per_seq'],
            seq_names=train_seqs,
        )

        if cfg['data'].get('coco_root'):
            coco = COCONegativeDataset(
                coco_root=cfg['data']['coco_root'],
                num_samples=cfg['data'].get('coco_samples', 50000)
            )
            train_dataset = ConcatDataset([train_dataset, coco])

        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg['train']['batch_size'],
            shuffle=True,
            num_workers=cfg['train']['num_workers'],
            collate_fn=collate_tracking,
            drop_last=True
        )

        val_dataset = UAVTrackingDataset(
            data_root=cfg['data']['uav123_root'],
            pairs_per_seq=cfg['data'].get('val_pairs_per_seq', 10),
            seq_names=val_seqs,
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=cfg['train']['batch_size'],
            shuffle=False,
            num_workers=cfg['train']['num_workers'],
            collate_fn=collate_tracking,
            drop_last=False
        )

        return train_loader"""

    assert old_build in c, "_build_dataloaders not found"
    c = c.replace(old_build, new_build)

    old_train = """    def train(self, checkpoint_dir='checkpoints'):
        os.makedirs(checkpoint_dir, exist_ok=True)

        loader = self._build_dataloaders()
        self.model.to(self.device)
        self.model.train()

        for epoch in range(self.start_epoch, self.cfg['train']['epochs']):
            if epoch < self.warmup_epochs:
                lr = self.cfg['train']['lr'] * (epoch + 1) / self.warmup_epochs
                for pg in self.optimizer.param_groups:
                    pg['lr'] = lr

            epoch_loss = 0.0
            epoch_cls = 0.0
            epoch_reg = 0.0

            pbar = tqdm(loader, desc=f'Epoch {epoch+1}')
            for batch in pbar:
                template = batch['template'].to(self.device)
                search = batch['search'].to(self.device)
                cls_target = batch['cls_target'].to(self.device)
                reg_target = batch['reg_target'].to(self.device)
                reg_mask = batch['reg_mask'].to(self.device)

                cls_pred, reg_pred = self.model(template, search)

                B, C, H, W = cls_pred.shape
                cls_target_flat = cls_target.permute(0, 2, 1, 3, 4)
                cls_target_flat = cls_target_flat.reshape(B, -1, H, W)

                B, C, H, W = reg_pred.shape
                reg_pred_flat = reg_pred.view(B, 5, 4, H, W).permute(0, 2, 1, 3, 4).reshape(B, 4, -1)

                total, cls_loss, reg_loss = self.criterion(
                    cls_pred, reg_pred_flat, cls_target_flat, reg_target, reg_mask
                )

                self.optimizer.zero_grad()
                total.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
                self.optimizer.step()

                epoch_loss += total.item()
                epoch_cls += cls_loss.item()
                epoch_reg += reg_loss.item()
                pbar.set_postfix(loss=total.item(), cls=cls_loss.item(), reg=reg_loss.item())

            if epoch >= self.warmup_epochs:
                self.scheduler.step()

            avg_loss = epoch_loss / len(loader)
            avg_cls = epoch_cls / len(loader)
            avg_reg = epoch_reg / len(loader)
            logger.info(f'Epoch {epoch+1}: loss={avg_loss:.4f} cls={avg_cls:.4f} reg={avg_reg:.4f}')

            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'loss': avg_loss,
                }, os.path.join(checkpoint_dir, 'best_model.pth'))
                logger.info(f'Checkpoint saved (loss={avg_loss:.4f})')

            if (epoch + 1) % self.cfg['train'].get('save_every', 10) == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'loss': avg_loss,
                }, os.path.join(checkpoint_dir, f'checkpoint_epoch{epoch+1}.pth'))

        torch.save({
            'epoch': self.cfg['train']['epochs'] - 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': avg_loss,
        }, os.path.join(checkpoint_dir, 'final_model.pth'))
        logger.info('Training complete.')"""

    new_train = """    def _validate(self):
        self.model.eval()
        total_loss = 0.0
        count = 0
        with torch.no_grad():
            for batch in self.val_loader:
                template = batch['template'].to(self.device)
                search = batch['search'].to(self.device)
                cls_target = batch['cls_target'].to(self.device)
                reg_target = batch['reg_target'].to(self.device)
                reg_mask = batch['reg_mask'].to(self.device)

                cls_pred, reg_pred = self.model(template, search)

                B, C, H, W = cls_pred.shape
                cls_target_flat = cls_target.permute(0, 2, 1, 3, 4).reshape(B, -1, H, W)
                reg_pred_flat = reg_pred.view(B, 5, 4, H, W).permute(0, 2, 1, 3, 4).reshape(B, 4, -1)

                total, _, _ = self.criterion(
                    cls_pred, reg_pred_flat, cls_target_flat, reg_target, reg_mask
                )
                total_loss += total.item()
                count += 1

        self.model.train()
        return total_loss / max(count, 1)

    def train(self, checkpoint_dir='checkpoints'):
        os.makedirs(checkpoint_dir, exist_ok=True)

        loader = self._build_dataloaders()
        self.model.to(self.device)
        self.model.train()

        logger.info(
            'Starting: epochs=%d, patience=%d, val_interval=%d',
            self.cfg['train']['epochs'], self.patience, self.val_interval
        )

        for epoch in range(self.start_epoch, self.cfg['train']['epochs']):
            if epoch < self.warmup_epochs:
                lr = self.cfg['train']['lr'] * (epoch + 1) / self.warmup_epochs
                for pg in self.optimizer.param_groups:
                    pg['lr'] = lr

            epoch_loss = 0.0
            epoch_cls = 0.0
            epoch_reg = 0.0

            pbar = tqdm(loader, desc=f'Epoch {epoch+1}')
            for batch in pbar:
                template = batch['template'].to(self.device)
                search = batch['search'].to(self.device)
                cls_target = batch['cls_target'].to(self.device)
                reg_target = batch['reg_target'].to(self.device)
                reg_mask = batch['reg_mask'].to(self.device)

                cls_pred, reg_pred = self.model(template, search)

                B, C, H, W = cls_pred.shape
                cls_target_flat = cls_target.permute(0, 2, 1, 3, 4)
                cls_target_flat = cls_target_flat.reshape(B, -1, H, W)

                B, C, H, W = reg_pred.shape
                reg_pred_flat = reg_pred.view(B, 5, 4, H, W).permute(0, 2, 1, 3, 4).reshape(B, 4, -1)

                total, cls_loss, reg_loss = self.criterion(
                    cls_pred, reg_pred_flat, cls_target_flat, reg_target, reg_mask
                )

                self.optimizer.zero_grad()
                total.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
                self.optimizer.step()

                epoch_loss += total.item()
                epoch_cls += cls_loss.item()
                epoch_reg += reg_loss.item()
                pbar.set_postfix(loss=total.item(), cls=cls_loss.item(), reg=reg_loss.item())

            if epoch >= self.warmup_epochs:
                self.scheduler.step()

            avg_loss = epoch_loss / len(loader)
            avg_cls = epoch_cls / len(loader)
            avg_reg = epoch_reg / len(loader)

            do_val = ((epoch + 1) % self.val_interval == 0)
            if do_val:
                val_loss = self._validate()
                logger.info(
                    'Epoch %d: train=%.4f cls=%.4f reg=%.4f val=%.4f',
                    epoch + 1, avg_loss, avg_cls, avg_reg, val_loss
                )

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.early_stop_counter = 0
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict(),
                        'loss': avg_loss,
                        'val_loss': val_loss,
                    }, os.path.join(checkpoint_dir, 'best_model.pth'))
                    logger.info('Best model saved (val_loss=%.4f)', val_loss)
                else:
                    self.early_stop_counter += 1
                    logger.info(
                        'EarlyStop %d/%d (best val=%.4f)',
                        self.early_stop_counter, self.patience, self.best_val_loss
                    )
                    if self.early_stop_counter >= self.patience:
                        logger.info('Early stopping at epoch %d', epoch + 1)
                        break
            else:
                logger.info('Epoch %d: loss=%.4f cls=%.4f reg=%.4f',
                            epoch + 1, avg_loss, avg_cls, avg_reg)

            if (epoch + 1) % self.cfg['train'].get('save_every', 10) == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'loss': avg_loss,
                }, os.path.join(checkpoint_dir, f'checkpoint_epoch{epoch+1}.pth'))

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': avg_loss,
            'val_loss': self.best_val_loss,
        }, os.path.join(checkpoint_dir, 'final_model.pth'))
        logger.info('Training done. Best val_loss=%.4f', self.best_val_loss)"""

    assert old_train in c, "train method not found"
    c = c.replace(old_train, new_train)

    with open(path, 'w') as f:
        f.write(c)
    print("[OK] trainer.py: validation + early stopping")


def fix_config():
    path = os.path.join(BASE, "configs", "default.yaml")
    with open(path) as f:
        c = f.read()
    c = c.replace("  epochs: 200", "  epochs: 80")
    c = c.replace(
        "  save_every: 10",
        "  save_every: 10\n  patience: 15\n  val_interval: 1\n  val_pairs_per_seq: 10"
    )
    with open(path, 'w') as f:
        f.write(c)
    print("[OK] configs/default.yaml: epochs=80, patience=15")


def fix_test_script():
    path = os.path.join(BASE, "scripts", "test.py")
    with open(path) as f:
        c = f.read()

    c = c.replace(
        "parser.add_argument('--checkpoint', type=str, required=True)",
        "parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth')"
    )

    old_debug = "parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')"
    new_debug = old_debug + "\n    parser.add_argument('--debug', action='store_true', help='Print per-frame debug info')"
    c = c.replace(old_debug, new_debug)

    old_print = "logger.info(f'Mean Precision (20px): {results[\"mean_precision\"]:.3f}')\n    logger.info(f'Mean Success (IoU=0.5): {results[\"mean_success\"]:.3f}')\n    logger.info(f'AUC: {results[\"auc\"]:.3f}')"
    new_print = old_print + """

    if args.debug and 'seq_results' in results:
        for seq_name, seq_res in list(results['seq_results'].items())[:3]:
            errors = seq_res.get('errors', [])
            overlaps = seq_res.get('overlaps', [])
            if errors:
                logger.info('Debug %s: frame0 err=%.1f iou=%.3f',
                            seq_name, errors[0], overlaps[0] if overlaps else -1)
            if len(errors) > 5:
                logger.info('Debug %s: frame5 err=%.1f iou=%.3f',
                            seq_name, errors[5], overlaps[5] if len(overlaps) > 5 else -1)"""

    c = c.replace(old_print, new_print)

    with open(path, 'w') as f:
        f.write(c)
    print("[OK] test.py: default checkpoint, --debug flag")


def fix_iou_threshold():
    """Lower positive IoU threshold from 0.6 → 0.5 so non-square targets get regression signal."""
    patched = False
    for fpath in [
        os.path.join(BASE, "src", "trainer.py"),
        os.path.join(BASE, "src", "data", "dataset.py"),
    ]:
        with open(fpath) as f:
            c = f.read()
        orig = c
        c = c.replace("if iou > 0.6:", "if iou > 0.5:")
        c = c.replace("if iou_map[i, j, k] > 0.6:", "if iou_map[i, j, k] > 0.5:")
        c = c.replace("iou_map > 0.6", "iou_map > 0.5")
        if c != orig:
            with open(fpath, 'w') as f:
                f.write(c)
            patched = True
    if patched:
        print("[OK] IoU threshold: 0.6 → 0.5")
    else:
        print("[OK] IoU threshold: already 0.5")


def _apply_patch(path, old, new, name):
    """Replace old with new if old is present; skip if new already present."""
    with open(path) as f:
        c = f.read()
    if old in c:
        c = c.replace(old, new)
        with open(path, 'w') as f:
            f.write(c)
        print(f"[OK] {name}")
        return True
    if new in c:
        print(f"[OK] {name} (already patched)")
        return True
    raise AssertionError(f"{name}: pattern not found in {path}")


def fix_anchor_centering():
    """Fix anchor centers: use j*stride + center_offset so they are centered in the 255x255 crop
    instead of shifted into the top-left quadrant (0..128)."""

    # --- dataset.py: _anchors() ---
    path = os.path.join(BASE, "src", "data", "dataset.py")
    _apply_patch(path,
        """def _anchors(stride, ratios, response_sz, scale=ANCHOR_SCALE):
    anchors = []
    for i in range(response_sz):
        for j in range(response_sz):
            cx = j * stride
            cy = i * stride""",
        """def _anchors(stride, ratios, response_sz, scale=ANCHOR_SCALE, search_sz=255):
    anchors = []
    center = (search_sz - (response_sz - 1) * stride) / 2.0
    for i in range(response_sz):
        for j in range(response_sz):
            cx = j * stride + center
            cy = i * stride + center""",
        "dataset.py: _anchors() centered")

    # --- dataset.py: _compute_targets() Gaussian label (dead code) ---
    with open(path) as f:
        c = f.read()
    old_g = """        gaussian = _gaussian_label((response_sz, response_sz),
                                    (cx_crop / STRIDE, cy_crop / STRIDE), sigma)"""
    new_g = """        center = (self.search_sz - (response_sz - 1) * STRIDE) / 2.0
        gx_resp = (cx_crop - center) / STRIDE
        gy_resp = (cy_crop - center) / STRIDE
        gaussian = _gaussian_label((response_sz, response_sz),
                                    (gx_resp, gy_resp), sigma)"""
    if old_g in c:
        c = c.replace(old_g, new_g)
        old_a = """        anc = _anchors(STRIDE, ANCHOR_RATIOS, response_sz, ANCHOR_SCALE)"""
        new_a = """        anc = _anchors(STRIDE, ANCHOR_RATIOS, response_sz, ANCHOR_SCALE, self.search_sz)"""
        _apply_patch(path, old_a, new_a,
                     "dataset.py: _compute_targets() anchors centered (dead code)")
        with open(path, 'w') as f:
            f.write(c)
        print("[OK] dataset.py: _compute_targets() gaussian centered (dead code)")
    elif new_g in c:
        print("[OK] dataset.py: _compute_targets() gaussian centered (dead code) (already patched)")
    else:
        print("[OK] dataset.py: _compute_targets() absent or skipped")

    # --- trainer.py: compute_targets() Gaussian label ---
    _apply_patch(
        os.path.join(BASE, "src", "trainer.py"),
        """    sigma = stride // 2
    gaussian = _gaussian_label((response_sz, response_sz),
                                (cx_crop / stride, cy_crop / stride), sigma)""",
        """    sigma = stride // 2
    center = (search_sz - (response_sz - 1) * stride) / 2.0
    gx_resp = (cx_crop - center) / stride
    gy_resp = (cy_crop - center) / stride
    gaussian = _gaussian_label((response_sz, response_sz),
                                (gx_resp, gy_resp), sigma)""",
        "trainer.py: compute_targets() gaussian centered")

    # --- trainer.py: _anchors() call ---
    _apply_patch(
        os.path.join(BASE, "src", "trainer.py"),
        """    anc = _anchors(stride, ANCHOR_RATIOS, response_sz, ANCHOR_SCALE)""",
        """    anc = _anchors(stride, ANCHOR_RATIOS, response_sz, ANCHOR_SCALE, search_sz)""",
        "trainer.py: compute_targets() anchors centered")

    # --- tracker.py: _build_anchors() ---
    _apply_patch(
        os.path.join(BASE, "src", "tracker.py"),
        """        anchors = []
        for i in range(self.response_sz):
            for j in range(self.response_sz):
                cx = j * self.stride
                cy = i * self.stride""",
        """        anchors = []
        center = (self.search_sz - (self.response_sz - 1) * self.stride) / 2.0
        for i in range(self.response_sz):
            for j in range(self.response_sz):
                cx = j * self.stride + center
                cy = i * self.stride + center""",
        "tracker.py: _build_anchors() centered")


if __name__ == "__main__":
    fix_anchor_centering()
    fix_train_arg()
    add_debug_to_compute_targets()
    create_split_module()
    fix_dataset_seq_names()
    rewrite_trainer()
    fix_config()
    fix_test_script()
    fix_iou_threshold()
    print("\n=== All fixes applied ===")
    print("Next: python /content/quack-track/scripts/train.py")
    print("      --config /content/quack-track/configs/default.yaml")
    print("      --checkpoint-dir /content/quack-track/checkpoints")
