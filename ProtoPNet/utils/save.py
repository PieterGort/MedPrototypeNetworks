import os
import torch
from PIL import Image
from argparse import ArgumentParser
import glob
import os

def save_model_w_condition(model, model_dir, model_name, metric, target_value, log=print):
    '''
    model: this is not the multigpu model
    '''
    if metric > target_value:
        log('\tabove {0:.2f}%'.format(target_value * 100))
        # torch.save(obj=model.state_dict(), f=os.path.join(model_dir, (model_name + '{0:.4f}.pth').format(accu)))
        torch.save(obj=model, f=os.path.join(model_dir, (model_name + '{0:.4f}.pth').format(metric)))

def save_best_model(model, model_dir, model_name, scores, current_epoch, after_epoch, log=print):
    """
    Saves the model based on the highest AUC-ROC and lowest cluster and separation costs, starting after a specific epoch.

    Args:
    - model (torch.nn.Module): The model to save.
    - model_dir (str): Directory to save the model.
    - model_name (str): Filename for the saved model.
    - scores (dict): Dictionary containing various metrics including AUC-ROC, cluster cost, and separation cost.
    - current_epoch (int): The current epoch number in the training process.
    - after_epoch (int): The epoch number after which the model starts saving.
    - log (function): Function used for logging messages.
    """
    if current_epoch <= after_epoch:
        log(f"Epoch {current_epoch} is before or equal to the specified after_epoch {after_epoch}. Skipping save.")
        return

    # Initialize best scores dictionary if not already existing
    if not hasattr(save_best_model, 'best_scores'):
        save_best_model.best_scores = {
            'AUROC': 0,  # Initialize to the lowest possible AUC-ROC
            # 'cluster_cost': float('inf'),  # Initialize to the highest possible cluster cost
            # 'separation_cost': float('inf')  # Initialize to the highest possible separation cost
        }

    current_AUROC = scores.get('AUROC', 0)
    current_cluster_cost = scores.get('cluster_cost', float('inf'))
    current_separation_cost = scores.get('separation_cost', float('inf'))

    # Check if current model is better based on AUC-ROC
    if (current_AUROC > save_best_model.best_scores['AUROC']):
        # current_cluster_cost < save_best_model.best_scores['cluster_cost'] and
        # current_separation_cost < save_best_model.best_scores['separation_cost']):

        save_best_model.best_scores['AUROC'] = current_AUROC
        # save_best_model.best_scores['cluster_cost'] = current_cluster_cost
        # save_best_model.best_scores['separation_cost'] = current_separation_cost

        # Save the model
        if not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, model_name)
        torch.save(model, f=os.path.join(model_dir, (model_name + '{0:.4f}.pth').format(current_AUROC)))
        log(f"Model saved at {model_path} with AUROC {current_AUROC}")
    else:
        log(f"Model not saved: current AUROC {current_AUROC} does not improve over best scores AUROC {save_best_model.best_scores['AUROC']}.")
    
def parse_args():
    parser = ArgumentParser(
        description="This script is convert png file to eps file")
    parser.add_argument('png_dir', help="Target image directory")

    args = parser.parse_args()
    return args.png_dir


def remove_transparency(im, bg_color=(255, 255, 255)):
    if im.mode in ('RGBA', 'LA') or (im.mode == 'P' and 'transparency' in im.info):
        alpha = im.convert('RGBA').split()[-1]

        bg = Image.new("RGBA", im.size, bg_color + (255,))
        bg.paste(im, mask=alpha)
        return bg
    else:
        return im


def convert(png_dir, eps_dir):
    """
    Converts the .png file to .eps file
    """
    img_list = glob.glob(png_dir+"/*.png")
    img_list += glob.glob(png_dir+"/*.jpg")
    
    os.makedirs(eps_dir, exist_ok=True)
    
    for img in img_list:
        im = Image.open(img)
        if im.mode in ('RGBA', 'LA'):
            im = remove_transparency(im)
            im = im.convert('RGB')
        name = os.path.splitext(img)[0].split('\\')[-1]
        im.save(eps_dir + '/' + name + ".eps", lossless=True)


def main():
    png_dir = r"C:\Users\PCpieter\OneDrive\Documents\MSc\MScThesis\Thesis\Figures\png_dir"
    #png_dir = r"C:\Users\piete\OneDrive\Documents\MSc\MScThesis\Thesis\Figures\png_dir"
    eps_dir = png_dir.replace("png_dir", "eps_dir")
    print(png_dir)
    print("Converting png to eps...")
    convert(png_dir, eps_dir)
    print("Conversion is finishing. \nOutput path is {}".format(eps_dir))


if __name__ == "__main__":
    main()

