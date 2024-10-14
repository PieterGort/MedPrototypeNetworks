import pandas as pd
import numpy as np
import os
import seaborn as sns
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import re
import matplotlib.pyplot as plt
import model

from dataset import MyDataset

# Function to extract features from a sentence
def extract_features(sentence, attributes):
    features = {key: 0 for key in attributes}
    for feature, keywords in attributes.items():
        for keyword in keywords:
            if keyword in sentence:
                features[feature] = 1
            else:
                features[feature] = 0
    return features


def create_features_csv(sentences_path, image_path, attributes):

    df = pd.read_csv(sentences_path)
    feature_vectors = []

    for sentence in df['sentence']:
        feature_vectors.append(extract_features(sentence, attributes))

    features_df = pd.DataFrame(feature_vectors)
    features_df["sentence"] = df["sentence"]
    features_df['filepath'] = df['path']
    features_df['filepath'] = features_df['filepath'].apply(lambda x: x.split("/")[-2:])
    features_df['filepath'] = features_df['filepath'].apply(lambda x: os.path.join(image_path, x[0], x[1]))
    features_df['target'] = df['label']
    features_df = features_df[features_df['filepath'].apply(os.path.exists)]
    features_df['feature_vector'] = features_df.apply(lambda row: np.array(row[list(attributes)].values, dtype=np.float32), axis=1)
    features_df.to_csv("exp/features.csv")

    return features_df, attributes

def load_model(net, path):
    checkpoint = torch.load(path, map_location='cuda:0')
    net.load_state_dict(checkpoint['model_state_dict'])
    return net

def run_inference(features_df, model_path):
        ## Forward the images through the model to obtain the minimum distances to prototypes
        preprocess = transforms.Compose([
        transforms.Resize((768, 768)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        dataset = MyDataset(features_df, preprocess)
        dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=5)

        # Load the model
        ppnet = model.construct_PPNet(base_architecture='resnet18',
                                    pretrained=True, img_size=(768, 768),
                                    prototype_shape=(20, 128, 1, 1),
                                    num_classes=2,
                                    prototype_activation_function='log',
                                    add_on_layers_type='regular',)
        
        ppnet = torch.nn.DataParallel(ppnet)
        ppnet = load_model(ppnet, model_path)
        device = torch.device('cuda')
        ppnet = ppnet.to(device)
        ppnet_multi = ppnet

        print(ppnet.module.last_layer.weight)
        print(ppnet.module.last_layer.weight.shape)
        # # Load the model
        # ppnet = torch.load(model_path)
        # ppnet = ppnet.cuda()
        # ppnet_multi = torch.nn.DataParallel(ppnet)

        all_prototype_activations = []

        length = len(dataloader)
        for i, (image, label) in enumerate(dataloader):
            image = image.cuda()
            label = label.cuda()
            with torch.no_grad():
                output_protopnet, min_distances, output_globalnet = ppnet_multi(image)
                prototype_activations = ppnet.module.distance_2_similarity(min_distances)
                all_prototype_activations.append(prototype_activations)
            print("processed batch ", i+1, " of ", length)
            
        all_prototype_activations = torch.cat(all_prototype_activations)
        print("The shape of all prototype activations:", all_prototype_activations.shape)
        all_prototype_activations = all_prototype_activations.cpu().numpy()

        #save the prototype activations to a numpy file
        np.save("exp/all_prototype_activations.npy", all_prototype_activations)

        return all_prototype_activations

def calculate_prototype_correlations(prototype_activations, word_embeddings, prototype_index):
    # initialize a list to hold the prototype activations for each word
    # Initialize a matrix of shape (num_images, num_words) to hold the prototype activations for each word

    num_images = word_embeddings.shape[0]
    num_words = word_embeddings.shape[1]

    prototype_correlations = np.zeros((num_images, num_words))

    for i in range(num_images):
        # Obtain the word embedding in image i
        word_embedding = word_embeddings[i, :]

        # Obtain the prototype activations for image i
        prototype_activation = prototype_activations[i, prototype_index]

        for j in range(num_words):
            if word_embedding[j] == 1:
                # Store the prototype activation for the present word
                prototype_correlations[i, j] = prototype_activation

    # save the prototype correlations to a numpy file
    #np.save("exp/prototype_correlations.npy", prototype_correlations)
    return prototype_correlations

def plot_boxplots(prototype_correlations, attribute_names, prototype_index, characteristic=None):

    colors = plt.cm.tab20(np.linspace(0, 1, len(attribute_names)))

    if characteristic == "size":
        attribute_indices = [0, 1, 2]
    elif characteristic == "shape":
        attribute_indices = [3, 4, 5, 6, 7, 8]
    elif characteristic == "surface":
        attribute_indices = [9, 10]
    elif characteristic == "depression":
        attribute_indices = [11]
    elif characteristic == "pit_pattern":
        attribute_indices = [12, 13, 14, 15, 16, 17]
    elif characteristic == "homogeneity":
        attribute_indices = [18, 19]
    elif characteristic == "vessels":
        attribute_indices = [20, 21, 22, 23]
    else:
        attribute_indices = list(range(len(attribute_names)))
    
    num_attributes = len(attribute_indices)

    fig, axes = plt.subplots(nrows=1, ncols=num_attributes, figsize=(5*num_attributes, 10), sharey=True)

    if num_attributes == 1:
        axes = [axes]

    for ax, attribute_index in zip(axes, attribute_indices):
        # Filter out zero values
        zero_filtered_prototype_correlations = prototype_correlations[prototype_correlations[:, attribute_index] != 0, attribute_index]

        # Create the boxplot
        box = ax.boxplot(zero_filtered_prototype_correlations, patch_artist=True, showfliers=False)

        # Set the box color
        for element in ['boxes', 'whiskers', 'caps']:
            plt.setp(box[element], color=colors[attribute_index])

        # Set face color
        for patch in box['boxes']:
            patch.set_facecolor(colors[attribute_index])

        # Set median line color and properties
        plt.setp(box['medians'], color='black', linewidth=2)

        # Set title and labels
        ax.set_title(attribute_names[attribute_index], fontsize=14, rotation=45)
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)

    fig.suptitle(f'Prototype {prototype_index} Activations for Different Attributes', fontsize=16, position=(0.5, 1.02))
    fig.text(0.04, 0.5, 'Similarity Score', va='center', rotation='vertical', fontsize=14)
    plt.xlabel('Attributes', fontsize=14)
    plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])
    plt.subplots_adjust(top=0.92)

    plt.savefig(f"exp/boxplots/prototype_{prototype_index}_activations.png")
    #plt.show()
    plt.close()

### MAIN ###
def main():
    sentences_path = r"C:\Users\piete\Documents\MScThesisLocal\GI_dataset\Colorectal_Polyps\polyp_sentences_train_no_comaBLI.csv"
    image_path = r'C:\Users\piete\Documents\MScThesisLocal\GI_dataset\Colorectal_Polyps\Polyp_DATASET_Portal_CZE_MUMC_clean_small_v2\train'
    model_path = r'C:\Users\piete\OneDrive\Documents\MSc\MScThesis\VSCodeThesis\BRAIxProtoPNet\braixprotopnet\saved_models\Polyp\resnet18\5fCV\net_trained_last_24_8'

    # sentences_path = r'c:\Users\PCpieter\Documents\vscode\mscthesis\Polyp\polyp_sentences_train_no_comaBLI.csv'
    # #sentences_path = r'c:\Users\PCpieter\Documents\vscode\mscthesis\Polyp\polyp_sentences_validation_no_comaBLI.csv'
    # image_path = r'c:\Users\PCpieter\Documents\vscode\mscthesis\Polyp\Polyp_DATASET_Portal_CZE_MUMC_clean_small_v2\train'
    # model_path = 'saved_models/Polyp/resnet18/5fCV/net_trained_last_24_8'

    attributes = {
        "morphology_protruded_pedunculated": ["protruded pendunculated"], # 0
        "morphology_protruded_sessile": ["protruded sessile"], # 1
        "morphology_superficial": ["superficial elevated"], # 2
        "morphology_flat": ["flat"], # 3
        "size_small": ["small"], # 4
        "size_diminutive": ["diminutive"], # 5
        "size_large": ["large"], # 6
        "mucus_yes": ["with mucus"],    # 7
        "mucus_no": ["without mucus"], # 8
        "regularity_regular": ["regular"], # 9
        "regularity_irregular": ["irregular"], # 10
        "depressed_pseudo": ["pseudo"], # 11
        "not_depressed": ["no depression"], # 12
        "features_yes": ["with features"], # 13
        "features_no": ["without features"], # 14
        "type_non_rounded": ["non-rounded"], # 15
        "type_round_not_dark": ["round not dark"], # 16
        "type_non_round_dark": ["round dark"], # 17
        "distribution_homogenous": ["homogenous"], # 18
        "distribution_heterogenous": ["heterogenous"], # 19
        "vessels_w_lacy": ["with lacy vessels"], # 20
        "vessels_with": ["with vessels"], # 21
        "vessels_without": ["without vessels"], # 22
        "vessels_non_continuous_pericryptal": ["with noncontinuous pericryptal vessels"], # 23
    }
        # Define mutually exclusive pairs
    mutually_exclusive_pairs = [
        ("with mucus", "without mucus"),
        ("protruded pendunculated", "protruded sessile", "superficial elevated", "flat"),
        ("small", "diminutive", "large"),
        ("regular", "irregular"),
        ("pseudo", "no depression"),
        ("with features", "without features"),
        ("non-rounded", "round not dark", "round dark"),
        ("homogenous", "heterogenous"),
        ("with lacy vessels", "with vessels", "without vessels", "with noncontinuous pericryptal vessels"),
    ]

    features_df, attributes = create_features_csv(sentences_path, image_path, attributes)

    if not os.path.exists("exp/all_prototype_activations.npy"):
                all_prototype_activations = run_inference(features_df, model_path)
    else:
        all_prototype_activations = np.load("exp/all_prototype_activations.npy")
    all_prototype_activations_pd = pd.DataFrame(all_prototype_activations)

    features_df['most_activated_prototype_idx'] = all_prototype_activations_pd.idxmax(axis=1)
    features_df['most_activated_prototype_activation'] = all_prototype_activations_pd.max(axis=1)
    word_embeddings = np.stack(features_df['feature_vector'].to_numpy())
    attribute_names = list(attributes.values())

    def min_max_normalize(S):
        S_norm = np.zeros_like(S)
        min_val = np.min(S)
        max_val = np.max(S)

        if max_val != min_val:
            S_norm = (S - min_val) / (max_val - min_val)
        else:
            S_norm = np.zeros_like(S)
        return S_norm

    def column_wise_normalization(S_transpose_W):
        S_transpose_W_norm = np.zeros_like(S_transpose_W)
        for i in range(S_transpose_W.shape[1]):
            column = S_transpose_W[:, i]
            col_min = np.min(column)
            col_max = np.max(column)
            
            if col_max != col_min:
                S_transpose_W_norm[:, i] = (column - col_min) / (col_max - col_min)
            else:
                S_transpose_W_norm[:, i] = 0
        return S_transpose_W_norm
    
    def row_wise_normalization(S_transpose_W):
        S_transpose_W_norm = np.zeros_like(S_transpose_W)
        for i in range(S_transpose_W.shape[0]):
            row = S_transpose_W[i, :]
            row_min = np.min(row)
            row_max = np.max(row)
            
            if row_max != row_min:
                S_transpose_W_norm[i, :] = (row - row_min) / (row_max - row_min)
            else:
                S_transpose_W_norm[i, :] = 0
        return S_transpose_W_norm
    
    def global_normalization(S_transpose_W):
        S_transpose_W_norm = np.zeros_like(S_transpose_W)
        global_min = np.min(S_transpose_W)
        global_max = np.max(S_transpose_W)

        if global_max != global_min:
            S_transpose_W_norm = (S_transpose_W - global_min) / (global_max - global_min)
        else:
            S_transpose_W_norm = np.zeros_like(S_transpose_W)
        return S_transpose_W_norm

    def z_score_normalization(S_transpose_W):
        S_transpose_W_norm = np.zeros_like(S_transpose_W)
        mean = np.mean(S_transpose_W)
        std = np.std(S_transpose_W)

        if std != 0:
            S_transpose_W_norm = (S_transpose_W - mean) / std
        else:
            S_transpose_W_norm = np.zeros_like(S_transpose_W)
        return S_transpose_W_norm
    
    # Copy the prototype activations matrix into matrix S and normalize it by column
    S = column_wise_normalization(all_prototype_activations) # shape [5565, 20]

    # Compute the word embeddings matrix
    W = word_embeddings
    S_transpose_W = np.dot(S.T, W)   # shape [5565, 20] x [20, 20] = [5565, 20]

    # NORMALIZATION TECHNIQUES, CHOOSE HERE:
    S_transpose_W = column_wise_normalization(S_transpose_W)

    def show_normalized_row(S_transpose_W, prototype_number, labels, norm_function):
        prototype_row = S_transpose_W[prototype_number, :]
        #norm_row = (prototype_row - np.min(prototype_row)) / (np.max(prototype_row) - np.min(prototype_row))
        norm_row = norm_function(prototype_row)
        plt.figure(figsize=(10, 2))
        sns.bartplot(x=norm_row, y=labels, palette="viridis")
        plt.xticks(np.arange(len(norm_row)), labels, rotation=45, ha='right')
        plt.yticks(f"{prototype_number}")
        plt.tight_layout()
        plt.savefig("exp/norm_row.png")
        plt.show()

    # color_mut_ex = sns.color_palette("dark", len(mutually_exclusive_pairs))
    # make it just black
    color_mut_ex = ["black"] * len(mutually_exclusive_pairs)
    word_colors = {}
    for i, pair in enumerate(mutually_exclusive_pairs):
        for word in pair:
            word_colors[word] = color_mut_ex[i]

    prototype_labels = [f'B {i}' if i < 10 else f'PM {i}' for i in range(S.shape[1])]
    word_labels = [item[0] for item in attributes.values()]  # Replace with actual word labels
    label_colors = [word_colors.get(word, "black") for word in word_labels]

    plt.figure(figsize=(20, 10))
    heatmap = sns.heatmap(S_transpose_W, cmap="YlGnBu", xticklabels=attribute_names, yticklabels=prototype_labels, annot=False, cbar=True)
    heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=45, ha='right', fontsize=20)
    heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0, fontsize=20)
    
    for tick_label, color in zip(heatmap.get_xticklabels(), label_colors):
        tick_label.set_color(color)

    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=20)
    plt.tight_layout()
    plt.savefig("exp/heatmap.png")
    plt.show()
    plt.close()

    # for p in range(20):
    #     # show_normalized_row(S_transpose_W, 16, word_labels, min_max_normalize)
    #     row = S_transpose_W[p, :]
    #     row = min_max_normalize(row)
    #     #print("The normalized row is: ", row)

    #     morphology_indices = [0, 1, 2, 3]
    #     size_indices = [4, 5, 6]
    #     mucus_indices = [7, 8]
    #     regularity_indices = [9, 10]
    #     depression_indices = [11, 12]
    #     features_indices = [13, 14]
    #     type_indices = [15, 16, 17]
    #     distribution_indices = [18, 19]
    #     vessels_indices = [20, 21, 22, 23]
        
    #     features = [size_indices, morphology_indices, mucus_indices, regularity_indices, depression_indices, features_indices, type_indices, distribution_indices, vessels_indices]

    #     all_present_features = []
    #     feature_vector_prototype = np.zeros(len(attributes))

    #     for feature in features:
    #         highest_value = 0
    #         highest_index = None
    #         present_feature = None
    #         for index in feature:
    #             if row[index] > highest_value:
    #                 highest_index = index
    #                 highest_value = row[index]
    #                 present_feature = attribute_names[index]
            
    #         if present_feature is not None:
    #             #print(f"Highest value for {present_feature}: {highest_value}")
    #             all_present_features.append(present_feature)


    #         feature_vector_prototype[highest_index] = 1

    #     df = features_df

    #     # Now, we check which sentences have a matching feature vector
    #     sentences_containing_words = []
    #     for i in range(len(df)):
    #         sentence_feature_vector = df['feature_vector'].iloc[i]

    #         # Compare the sentence feature vector with the prototype's feature vector
    #         if np.array_equal(sentence_feature_vector, feature_vector_prototype):
    #             sentences_containing_words.append(df.iloc[i, 0])  # Add the sentence if the feature vector matches

    #     # Print the results
    #     print("Number of sentences containing all features: ", len(sentences_containing_words))
    #     w = all_present_features

    #     matching_counts = []
    #     for i in range(len(df)):
    #         sentence_feature_vector = df['feature_vector'].iloc[i]

    #         matching_features = np.sum(np.logical_and(sentence_feature_vector, feature_vector_prototype))
    #         total_prototype_features = np.sum(feature_vector_prototype)

    #         matching_counts.append(matching_features / total_prototype_features)
        
    #     df['matching_score'] = matching_counts
    #     #print(df[['sentence', 'matching_score']])
    #     print(f"The maximum matching score for prototype {p} is:", df['matching_score'].max())
    #     # print the indexes of the sentences with the highest matching score
    #     # print(len(df['sentence'][df['matching_score'] == df['matching_score'].max()]))

    #     # most_matching_samples = df[df['matching_score'] == df['matching_score'].max()]

    #     # from scipy.stats import pearsonr
    #     # pearson_corr, p_value = pearsonr(most_matching_samples['target'], most_matching_samples['matching_score'])
    #     # print(f"Pearson Correlation: {pearson_corr}, P-Value: {p_value}")   

    #     #print(f"Prototype Number {p}: a {w[0]} polyp {w[1]}.surface {w[2]} {w[3]} {w[4]}.pits {w[5]} {w[6]} {w[7]}.{w[8]}")

    # print("Done")

    # # Load the dataset
    # df = pd.read_csv(sentences_path)

    # # Filter sentences based on all the features
    # features_to_check = [item for sublist in all_present_features for item in sublist]

    # sentences = df['sentence']
    # sentences_containing_words = []
    # for i in range(len(sentences)):
    #     if all(word in sentences[i] for word in features_to_check):
    #         sentences_containing_words.append(sentences[i])
    #     else:
    #         continue
    
    # #print("Number of sentences containing all features: ", len(sentences_containing_words))
    # w = [item for sublist in all_present_features for item in sublist]
    # print(f"Prototype Number {p}: a {w[0]} polyp {w[1]}.surface {w[2]} {w[3]} {w[4]}.pits {w[5]} {w[6]} {w[7]}.{w[8]}")

    # print("Done")


        # if all(word in df['sentence'].values for word in features_to_check):
        #     sentences = df[df['sentence'].str.contains(all_present_features[0])]
        #     for feature in all_present_features[1:]:
        #         sentences = sentences[sentences['sentence'].str.contains(feature)]
        #     print(len(sentences))
        #     print(sentences['label'].value_counts())
        # else:
        #     print("No sentences found")
  
    # # check all sentences containing "small"
    # sentences = df[df['sentence'].str.contains(all_present_features[1])]
    # print(len(sentences))
    # # check "sentences" for "protruded sessile"
    # sentences = sentences[sentences['sentence'].str.contains(all_present_features[0])]
    # print(len(sentences))
    # # check "sentences" for "with mucus"
    # sentences = sentences[sentences['sentence'].str.contains("with mucus")]
    # print(len(sentences))
    # # check "sentences" for "regular"
    # sentences = sentences[sentences['sentence'].str.contains("regular")]
    # print(len(sentences))
    # # check "sentences" for "no depression"
    # sentences = sentences[sentences['sentence'].str.contains("no depression")]
    # print(len(sentences))
    # # check "sentences" for "with features"
    # sentences = sentences[sentences['sentence'].str.contains("with features")]
    # print(len(sentences))
    # # check for "round not dark"
    # sentences = sentences[sentences['sentence'].str.contains("non-rounded")]
    # print(len(sentences))
    # # check for "homogenous"
    # sentences = sentences[sentences['sentence'].str.contains("heterogenous")]
    # print(len(sentences))
    # # check for "without vessels"
    # sentences = sentences[sentences['sentence'].str.contains("with noncontinuous pericryptal vessels")]
    # print(len(sentences))
    
    # print(sentences['label'].value_counts())

if __name__ == "__main__":
    main()


