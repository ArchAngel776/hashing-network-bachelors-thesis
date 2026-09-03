def class_collisions(hashes, labels):
    _, indices = hashes.unique(sorted=False, return_inverse=True, dim=0)

    collided_codes  = 0
    collided_images = 0

    for index in indices.unique():
        group_labels = labels[indices == index]

        if group_labels.unique().numel() > 1:
            collided_codes  += 1
            collided_images += group_labels.numel()

    return collided_codes, collided_images
