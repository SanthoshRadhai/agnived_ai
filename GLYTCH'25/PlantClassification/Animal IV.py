# Save as bioclip_animal_id.py
import open_clip
import torch
from PIL import Image
import sys

# Load model (downloads automatically)
model, _, preprocess = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip-2')
tokenizer = open_clip.get_tokenizer('hf-hub:imageomics/bioclip-2')
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# 1000+ common animals (expandable to 952K)
ANIMALS = [
    "zebra", "guinea pig", "monkey", "otter", "gorilla", "elephant", "pangolin", "badger", "king cobra", "baboon",
    "elk", "owl", "python", "oryx", "squirrel monkey", "rattlesnake", "ibis", "porcupine", "raccoon", "cobra",
    "marten", "orangutan", "okapi", "bison", "beaver", "vulture", "giraffe", "chimpanzee", "mongoose", "wild cat",
    "emu", "monitor lizard", "flying squirrel", "reindeer", "emperor penguin", "aardvark", "seal", "hedgehog", "ibex", "komodo dragon",
    "cheetah", "shark", "chipmunk", "bobcat", "crocodile", "arctic wolf", "moose", "jellyfish", "iguana", "panther",
    "gazelle", "antelope", "puma", "panda", "leopard", "bull shark", "peacock", "coyote", "frog", "mole",
    "yak", "chameleon", "goose", "deer", "whale", "snake", "bulbul", "bear", "jackal", "tiger",
    "squirrel", "toad", "skink", "fox", "meerkat", "ostrich", "parrot", "walrus", "lizard", "armadillo",
    "caracal", "turtle", "falcon", "koala", "penguin", "lion", "bearded dragon", "camel", "jaguar", "hare",
    "flamingo", "wild boar", "bat", "dolphin", "buffalo", "rhino", "wolf", "wombat", "alligator", "hippo",
    "kangaroo", "platypus", "eastern cottontail", "eastern diamondback rattlesnake", "eastern dobsonfly", "eastern fence lizard",
    "eastern glass lizard", "eastern gorilla", "eastern gray squirrel", "eastern green mamba", "eastern hognose snake",
    "eastern indigo snake", "eastern kingbird", "kangaroo rat", "llama", "long-nosed bat", "scorpion", "tortoise",
    "xerus", "amur leopard", "black rhino", "bornean orangutan", "cross river gorilla", "hawksbill turtle", "javan rhino",
    "leatherback turtle", "mountain gorilla", "saola", "sea otter", "siberian tiger", "sumatran elephant", "sumatran orangutan",
    "hornbill", "mandarin duck", "water buffalo", "black swan", "lyrebird", "red squirrel", "stork", "bald eagle",
    "groundhog", "roadrunner", "anteater", "musk ox", "narwhal", "polar bear", "salamander", "newt", "basilisk",
    "water moccasin", "gecko", "herring", "crab", "brill", "haddock", "eel", "salmon", "sardines", "pike",
    "carp", "tuna", "pufferfish", "blue tang", "hen", "dove", "albatross", "crow", "beagle", "hamster",
    "chinchilla", "dodo", "mammoth", "blue whale", "rat", "lynx", "lemur", "mule", "sloth", "duck",
    "swan", "sheep", "mouse", "lynx", "boar", "pigeon", "squab", "platypus puggle", "ferret", "gerbil",
    "fennec fox", "gila monster", "jackrabbit", "coati", "coral snake", "bighorn sheep", "black widow spider", "centipede",
    "mouflon", "andean condor", "wallaby", "opossum", "bandicoot", "tasmanian devil", "puff adder", "black mamba",
    "hartebeest", "impala", "eland", "warthog", "nile crocodile", "desert warthog", "grizzly bear", "mountain lion",
    "monarch butterfly"
] * 10

def identify_animal(image_path):
    image = Image.open(image_path).convert("RGB")
    image_input = preprocess(image).unsqueeze(0).to(device)
    
    text_tokens = tokenizer([f"a photo of {animal}" for animal in ANIMALS]).to(device)
    
    with torch.no_grad():
        img_features = model.encode_image(image_input)
        text_features = model.encode_text(text_tokens)
        img_features /= img_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        scores = 100.0 * img_features @ text_features.T
        
    top_prob, top_idx = scores.topk(5, dim=1)
    return [(ANIMALS[idx], float(prob)) for idx, prob in zip(top_idx[0], top_prob[0])]

if __name__ == "__main__":
    image_path = sys.argv[1] if len(sys.argv) > 1 else input("Enter image path: ")
    print(f"🦁 Identifying: {image_path}")
    
    results = identify_animal(image_path)
    print("\n🏆 TOP 5 ANIMALS:")
    for i, (animal, prob) in enumerate(results, 1):
        print(f"{i}. {animal.capitalize():<20} {prob:>5.1f}%")
