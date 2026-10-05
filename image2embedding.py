import torch
# import torch.nn.parallel
import torchvision.transforms as transforms
from trijoint import im2recipe
from checkpoint_io import load_checkpoint
import pickle
from args import get_parser
from PIL import Image
import os

# =============================================================================
parser = get_parser()
opts = parser.parse_args()
# =============================================================================

def get_device(opts):
    if not torch.cuda.device_count():
        return torch.device('cpu', 0)
    torch.cuda.manual_seed(opts.seed)
    return torch.device('cuda', 0)

def norm(input, p=2, dim=1, eps=1e-12):
    return input / input.norm(p,dim,keepdim=True).clamp(min=eps).expand_as(input)

def main(opts=None):
    if opts is None:
        opts = get_parser().parse_args()
    device = get_device(opts)

    im_path = opts.test_image_path
    ext = os.path.basename(im_path).split('.')[-1]
    if ext not in ['jpeg','jpg','png']:
        raise Exception("Wrong image format.")

    # create model
    model = im2recipe(opts, pretrained=False)  # the checkpoint supplies every weight
    model.visionMLP = torch.nn.DataParallel(model.visionMLP)
    model.to(device)

    # load checkpoint
    print("=> loading checkpoint '{}'".format(opts.model_path))
    checkpoint = load_checkpoint(opts.model_path, device, opts.trust_checkpoint)
    opts.start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' (epoch {})"
          .format(opts.model_path, checkpoint['epoch']))

    # data preparation, loaders
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
                transforms.Resize(256), # rescale the image keeping the original aspect ratio
                transforms.CenterCrop(224), # we get only the center of that rescaled
                transforms.ToTensor(),
                normalize])

    # load image
    im = Image.open(im_path).convert('RGB')
    im = transform(im)
    im = im.view((1,)+im.shape)
    # get model output
    output = model.visionMLP(im)
    output = norm(output)
    output = output.data.cpu().numpy()
    # save output
    with open(im_path.replace(ext,'pkl'), 'wb') as f:
        pickle.dump(output, f)

if __name__ == '__main__':
    main()
