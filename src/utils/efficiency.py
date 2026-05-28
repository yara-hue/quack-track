import torch


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def total_parameters(model):
    return sum(p.numel() for p in model.parameters())


def profile_model(model, template_size=127, search_size=255):
    try:
        from thop import profile
        template = torch.randn(1, 3, template_size, template_size)
        search = torch.randn(1, 3, search_size, search_size)
        flops, params = profile(model, inputs=(template, search), verbose=False)
        return flops, params
    except ImportError:
        return 0, 0


def measure_latency(model, template_size=127, search_size=255, device='cpu', num_warmup=10, num_iters=100):
    model.eval()
    model.to(device)
    template = torch.randn(1, 3, template_size, template_size).to(device)
    search = torch.randn(1, 3, search_size, search_size).to(device)

    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(template, search)

        if device == 'cuda':
            torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        if device == 'cuda':
            start.record()
        else:
            start = time.time()

        for _ in range(num_iters):
            _ = model(template, search)

        if device == 'cuda':
            end.record()
            torch.cuda.synchronize()
            elapsed = start.elapsed_time(end) / num_iters
        else:
            elapsed = (time.time() - start) * 1000 / num_iters

    return elapsed
