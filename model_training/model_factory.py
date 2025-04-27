from models.cnn import DogBehaviorCNN
from models.lstm import DogBehaviorLSTM
from models.transformer_fft import DogBehaviorTransformer
import config
from models.transformer_flash import DogBehaviorTransformerFlash
from models.bilstm import DogBehaviorBiLSTM
from models.bilstm_fusion import DogBehaviorBiLSTM_Fusion

def get_model(model_name, input_channels, num_classes):
    if model_name == "cnn":
        return DogBehaviorCNN(input_channels=input_channels, num_classes=num_classes)
    elif model_name == "lstm":
        return DogBehaviorLSTM(input_channels=input_channels, num_classes=num_classes)

    elif model_name == "bilstm":
        return DogBehaviorBiLSTM(input_channels=input_channels, num_classes=num_classes)

    elif model_name == "bilstm_fusion":
        return DogBehaviorBiLSTM_Fusion(input_channels=input_channels, num_classes=num_classes)

    elif model_name == "transformer":
        return DogBehaviorTransformer(
            input_channels=input_channels,
            seq_len=config.WINDOW_SIZE + 1,
            num_classes=num_classes
        )
    elif model_name == "transformer_flash":
        return DogBehaviorTransformerFlash(
            input_channels=input_channels,
            seq_len=config.WINDOW_SIZE + 1,
            num_classes=num_classes
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")

