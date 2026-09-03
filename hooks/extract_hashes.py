from torch import no_grad, cat


@no_grad()
def extract_hashes(model, source, device):
    model.eval()

    hashes = []
    labels = []

    for batch_images, batch_labels in source:
        image_hashes = model.generate(batch_images.to(device))

        hashes.append(image_hashes.cpu())
        labels.append(batch_labels.cpu())

    return cat(hashes, dim=0), cat(labels, dim=0)
