"""Some utilities."""
import torch

def seqs2batch(data):
    """Get a list of images from a list of sequences.

    Args:
        data: list of sequences (shaped batch_size x seq_len, with seq_len variable).

    Returns:
        images: list of images.
        seq_lens: list of sequence lengths.
        lookup_table: list (shaped batch_size x seq_len, with seq_len variable) containing
            the indices of images in the image list.

    """
    # Get all inputs and keep the information about the sequence they belong to.
    images = torch.Tensor()
    img_data = [i['images'] for i in data]
    seq_lens = torch.zeros(len(img_data)).int()
    lookup_table = []
    count = 0
    for seq_tag, seq_imgs in enumerate(img_data):
        seq_lookup = []
        for img in seq_imgs:
            images = torch.cat((images, img.unsqueeze(0)))
            seq_lookup.append(count)
            count += 1
            seq_lens[seq_tag] += 1
        lookup_table.append(seq_lookup)

    return images, seq_lens, lookup_table
