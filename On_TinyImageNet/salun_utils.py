"""Reusable SalUn primitives for TinyImageNet unlearning experiments."""

import random

import numpy as np
import torch


def generate_salun_weight_mask(model, forget_loader, loss_fn, args,
                               verbose=True):
    """Build the SalUn weight-saliency mask from forget-set gradients."""
    gradients = {
        name: torch.zeros_like(param, device=args.device)
        for name, param in model.named_parameters()
    }

    model.eval()
    for x, y in forget_loader:
        x, y = x.to(args.device), y.to(args.device)
        if y.ndim == 2:
            y = y.argmax(dim=1)

        logits_y = model(x)
        # Match the reference SalUn implementation: accumulate -CE gradients.
        loss = -loss_fn(logits_y, y)
        model.zero_grad(set_to_none=True)
        loss.backward()

        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.grad is not None:
                    gradients[name].add_(param.grad.detach())

    with torch.no_grad():
        grad_tensors = [
            grad.abs().flatten()
            for grad in gradients.values()
            if grad.numel() > 0
        ]
        if not grad_tensors:
            return {
                name: torch.zeros_like(param, device=args.device)
                for name, param in model.named_parameters()
            }

        all_scores = torch.cat(grad_tensors)
        mask_ratio = min(max(float(args.salun_mask_ratio), 0.0), 1.0)
        threshold_index = int(all_scores.numel() * mask_ratio)
        if threshold_index <= 0:
            saliency_threshold = torch.inf
        elif threshold_index >= all_scores.numel():
            saliency_threshold = all_scores.min() - 1
        else:
            saliency_threshold = torch.topk(
                all_scores, threshold_index, largest=True
            ).values[-1]

        mask = {}
        selected = 0
        total = 0
        for name, grad in gradients.items():
            mask_tensor = (grad.abs() >= saliency_threshold).to(
                dtype=grad.dtype,
                device=args.device,
            )
            mask[name] = mask_tensor
            selected += int(mask_tensor.sum().item())
            total += mask_tensor.numel()

    if verbose:
        print(
            f'SalUn mask generated: selected={selected}/{total} '
            f'({selected / max(total, 1):.4f}), '
            f'target_ratio={args.salun_mask_ratio:.4f}'
        )
    return mask


def _apply_salun_mask_to_grads(model, mask):
    for name, param in model.named_parameters():
        if param.grad is not None and name in mask:
            param.grad.mul_(
                mask[name].to(
                    device=param.grad.device,
                    dtype=param.grad.dtype,
                )
            )


def _restore_salun_masked_params(model, mask, theta0, optimizer):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name not in mask:
                continue
            mask_tensor = mask[name].to(
                device=param.device,
                dtype=param.dtype,
            )
            inv_mask = 1.0 - mask_tensor
            if torch.count_nonzero(inv_mask) > 0:
                param.data.mul_(mask_tensor).add_(
                    theta0[name].to(param.device) * inv_mask
                )

            state = optimizer.state.get(param, None)
            if state is not None and 'exp_avg' in state:
                state['exp_avg'].mul_(mask_tensor)
            if state is not None and 'exp_avg_sq' in state:
                state['exp_avg_sq'].mul_(mask_tensor)
            if state is not None and 'momentum_buffer' in state:
                state['momentum_buffer'].mul_(mask_tensor)


def salun_unlearn(forget_loader, retain_loader, model, loss_fn, args, mask,
                  verbose=True):
    """Apply random-label forgetting and masked retain-set fine-tuning."""
    optimizer = torch.optim.Adam(model.parameters(), lr=args.salun_lr)
    theta0 = {
        name: param.detach().clone()
        for name, param in model.named_parameters()
        if name in mask
    }

    if verbose:
        print('SalUn forget samples:', len(forget_loader.dataset))
    for epoch in range(args.unlearning_epochs):
        model.train()

        for x_f, y_f in forget_loader:
            x_f, y_f = x_f.to(args.device), y_f.to(args.device)
            if y_f.ndim == 2:
                y_f = y_f.argmax(dim=1)
            random_y = torch.randint(
                0,
                args.num_classes,
                y_f.shape,
                device=args.device,
            )

            loss = loss_fn(model(x_f), random_y)
            optimizer.zero_grad()
            loss.backward()
            _apply_salun_mask_to_grads(model, mask)
            optimizer.step()
            _restore_salun_masked_params(model, mask, theta0, optimizer)

        for x_r, y_r in retain_loader:
            x_r, y_r = x_r.to(args.device), y_r.to(args.device)
            if y_r.ndim == 2:
                y_r = y_r.argmax(dim=1)

            loss = loss_fn(model(x_r), y_r)
            optimizer.zero_grad()
            loss.backward()
            _apply_salun_mask_to_grads(model, mask)
            optimizer.step()
            _restore_salun_masked_params(model, mask, theta0, optimizer)

        if verbose:
            print(
                f'SalUn unlearning epoch '
                f'{epoch + 1}/{args.unlearning_epochs}'
            )
    return model


def set_random_seed(seed):
    """Set the deterministic RNG state used by paired SalUn worlds."""
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
