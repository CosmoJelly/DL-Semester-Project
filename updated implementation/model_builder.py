# model_builder.py
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from config import IMG_SIZE, LEARNING_RATE

# ECA (Efficient Channel Attention) block
class ECABlock(layers.Layer):
    """Efficient Channel Attention block for feature enhancement."""
    def __init__(self, kernel_size=3, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
    
    def build(self, input_shape):
        channels = input_shape[-1]
        self.avg_pool = layers.GlobalAveragePooling2D()
        self.conv = layers.Conv1D(1, kernel_size=self.kernel_size, padding='same', use_bias=False)
        self.sigmoid = layers.Activation('sigmoid')
    
    def call(self, inputs):
        # Global average pooling
        x = self.avg_pool(inputs)  # [batch, channels]
        x = tf.expand_dims(x, axis=1)  # [batch, 1, channels]
        # 1D convolution
        x = self.conv(x)  # [batch, 1, channels]
        x = self.sigmoid(x)
        x = tf.squeeze(x, axis=1)  # [batch, channels]
        x = tf.expand_dims(tf.expand_dims(x, axis=1), axis=1)  # [batch, 1, 1, channels]
        return inputs * x
    
    def get_config(self):
        config = super().get_config()
        config.update({"kernel_size": self.kernel_size})
        return config

# GAM (Global Attention Mechanism) block - inspired by Pacal 2024
class GAMBlock(layers.Layer):
    """Global Attention Mechanism combining channel and spatial attention."""
    def __init__(self, reduction=4, **kwargs):
        super().__init__(**kwargs)
        self.reduction = reduction
    
    def build(self, input_shape):
        channels = input_shape[-1]
        mid_channels = max(channels // self.reduction, 1)
        
        # Channel attention branch
        self.channel_avg = layers.GlobalAveragePooling2D()
        self.channel_max = layers.GlobalMaxPooling2D()
        self.channel_fc1 = layers.Dense(mid_channels, activation='relu')
        self.channel_fc2 = layers.Dense(channels, activation='sigmoid')
        
        # Spatial attention branch
        self.spatial_conv = layers.Conv2D(1, kernel_size=7, padding='same', activation='sigmoid')
    
    def call(self, inputs):
        # Channel attention
        avg_out = self.channel_avg(inputs)
        max_out = self.channel_max(inputs)
        channel_out = self.channel_fc2(self.channel_fc1(avg_out + max_out))
        channel_out = tf.expand_dims(tf.expand_dims(channel_out, axis=1), axis=1)
        x = inputs * channel_out
        
        # Spatial attention
        spatial_out = self.spatial_conv(x)
        return x * spatial_out
    
    def get_config(self):
        config = super().get_config()
        config.update({"reduction": self.reduction})
        return config

# Multi-scale Feature Fusion block
class MultiScaleFusion(layers.Layer):
    """Fuse features from multiple scales using attention."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def build(self, input_shapes):
        # input_shapes is a list of shapes from different layers
        self.num_scales = len(input_shapes)
        self.attention_weights = []
        for i in range(self.num_scales):
            self.attention_weights.append(self.add_weight(
                name=f'attention_weight_{i}',
                shape=(1,),
                initializer='ones',
                trainable=True
            ))
    
    def call(self, inputs):
        # inputs is a list of feature maps from different layers
        # Normalize attention weights
        weights = tf.stack([w for w in self.attention_weights])
        weights = tf.nn.softmax(weights)
        
        # Resize all to same size (use smallest)
        target_shape = tf.shape(inputs[0])[1:3]
        resized = []
        for feat in inputs:
            if tf.shape(feat)[1:3] != target_shape:
                feat = tf.image.resize(feat, target_shape)
            resized.append(feat)
        
        # Weighted fusion
        fused = sum(w * feat for w, feat in zip(weights, resized))
        return fused
    
    def get_config(self):
        config = super().get_config()
        return config

def get_base_model(name):
    size = (IMG_SIZE[0], IMG_SIZE[1], 3)
    name_lower = name.lower()
    
    # Standard CNN models
    if name_lower == "densenet201":
        return tf.keras.applications.DenseNet201(weights="imagenet", include_top=False, input_shape=size)
    if name_lower == "densenet121":
        return tf.keras.applications.DenseNet121(weights="imagenet", include_top=False, input_shape=size)
    if name_lower == "inceptionresnetv2":
        return tf.keras.applications.InceptionResNetV2(weights="imagenet", include_top=False, input_shape=size)
    if name_lower == "inceptionv3":
        return tf.keras.applications.InceptionV3(weights="imagenet", include_top=False, input_shape=size)
    if name_lower == "mobilenetv2":
        return tf.keras.applications.MobileNetV2(weights="imagenet", include_top=False, input_shape=size)
    if name_lower == "nasnetlarge":
        return tf.keras.applications.NASNetLarge(weights="imagenet", include_top=False, input_shape=size)
    if name_lower == "nasnetmobile":
        return tf.keras.applications.NASNetMobile(weights="imagenet", include_top=False, input_shape=size)
    if name_lower == "resnet152v2":
        return tf.keras.applications.ResNet152V2(weights="imagenet", include_top=False, input_shape=size)
    if name_lower == "vgg19":
        return tf.keras.applications.VGG19(weights="imagenet", include_top=False, input_shape=size)
    if name_lower == "xception":
        return tf.keras.applications.Xception(weights="imagenet", include_top=False, input_shape=size)
    
    # Vision Transformer (MaxViT) - try keras_cv, fallback to ViT
    if name_lower == "maxvit":
        try:
            import keras_cv
            # MaxViT from keras_cv
            try:
                return keras_cv.models.MaxViTBackbone.from_preset(
                    "maxvit_tiny_tf_224", input_shape=size
                )
            except (AttributeError, ValueError) as e:
                print(f"[WARN] MaxViT preset failed: {e}. Trying alternative...")
                # Try building MaxViT manually if preset fails
                raise ImportError("MaxViT preset not available")
        except ImportError:
            try:
                # Fallback to standard ViT if available
                from tensorflow.keras.applications import vit
                print("[INFO] Using standard ViT instead of MaxViT")
                return vit.ViTBase16(weights="imagenet", include_top=False, input_shape=size)
            except (ImportError, AttributeError):
                # If neither available, use a simple ViT-like architecture
                print("[WARN] keras_cv and ViT not available. Using simple ViT implementation.")
                print("[INFO] For better MaxViT support, install: pip install keras-cv")
                return build_simple_vit_backbone(size)
    
    raise ValueError(f"Unsupported model name: {name}. Supported: {', '.join(['DenseNet121', 'DenseNet201', 'InceptionV3', 'InceptionResNetV2', 'MobileNetV2', 'NasNetLarge', 'NasNetMobile', 'ResNet152V2', 'VGG19', 'Xception', 'MaxViT'])}")

def build_simple_vit_backbone(input_shape):
    """Simple ViT-like backbone as fallback."""
    from tensorflow.keras import layers, models
    
    inputs = layers.Input(shape=input_shape)
    
    # Patch embedding
    patch_size = 16
    num_patches = (input_shape[0] // patch_size) * (input_shape[1] // patch_size)
    embed_dim = 768
    
    # Reshape to patches
    x = layers.Conv2D(embed_dim, kernel_size=patch_size, strides=patch_size, padding='valid')(inputs)
    x = layers.Reshape((num_patches, embed_dim))(x)
    
    # Add positional embedding
    pos_embed = layers.Embedding(input_dim=num_patches, output_dim=embed_dim)(tf.range(num_patches))
    x = x + pos_embed
    
    # Transformer blocks (simplified)
    for _ in range(6):  # 6 transformer blocks
        # Multi-head self-attention
        attn_output = layers.MultiHeadAttention(num_heads=12, key_dim=64)(x, x)
        x = layers.Add()([x, attn_output])
        x = layers.LayerNormalization()(x)
        
        # Feed-forward
        ffn = layers.Dense(embed_dim * 4, activation='gelu')(x)
        ffn = layers.Dense(embed_dim)(ffn)
        x = layers.Add()([x, ffn])
        x = layers.LayerNormalization()(x)
    
    # Global average pooling
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Reshape((1, 1, embed_dim))(x)
    
    return models.Model(inputs, x, name='simple_vit_backbone')

def build_model(name, num_classes, base_trainable=False, dropout=0.4, use_attention=False, attention_type='eca'):
    """
    Build model with optional attention mechanism.
    
    Args:
        name: Model architecture name (can include _Attention or _GAM suffix)
        num_classes: Number of output classes
        base_trainable: Whether base model is trainable
        dropout: Dropout rate
        use_attention: If True, add attention block
        attention_type: 'eca' or 'gam' - type of attention to use
    """
    # Handle special model names
    base_name = name
    if name.endswith("_Attention"):
        base_name = name.replace("_Attention", "")
        use_attention = True
        attention_type = 'eca'
    elif name.endswith("_GAM"):
        base_name = name.replace("_GAM", "")
        use_attention = True
        attention_type = 'gam'
    elif name.endswith("_Hybrid"):
        # Hybrid: Attention + Contour Fusion
        base_name = name.replace("_Hybrid", "")
        # This will be handled separately
        pass
    
    base = get_base_model(base_name)
    base.trainable = base_trainable
    x = base.output
    
    # Handle different output shapes
    # ViT models might output different shapes, CNN models output [batch, H, W, C]
    if len(base.output.shape) == 4:  # CNN: [batch, H, W, C]
        # Add attention if requested (before pooling)
        if use_attention:
            if attention_type == 'eca' and base_name.lower() == "densenet121":
                x = ECABlock(kernel_size=3, name="eca_attention")(x)
            elif attention_type == 'gam':
                x = GAMBlock(reduction=4, name="gam_attention")(x)
        x = layers.GlobalAveragePooling2D()(x)
    elif len(base.output.shape) == 2:  # Already pooled (some ViT models)
        x = base.output
    else:
        # Fallback: try to pool
        x = layers.GlobalAveragePooling2D()(x) if len(x.shape) == 4 else layers.Flatten()(x)
    
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax", dtype="float32")(x)
    model = models.Model(base.input, outputs)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def build_model_with_contour_fusion(name, num_classes, base_trainable=False, dropout=0.4, 
                                   contour_feature_dim=4, use_attention=False):
    """
    Build model that fuses CNN features with contour features.
    
    Args:
        name: Model architecture name
        num_classes: Number of output classes
        base_trainable: Whether base model is trainable
        dropout: Dropout rate
        contour_feature_dim: Dimension of contour features (perimeter, area, epsilon, num_vertices)
        use_attention: If True, add attention before fusion (for DenseNet121)
    """
    base = get_base_model(name)
    base.trainable = base_trainable
    
    # CNN branch with optional attention
    cnn_features = base.output
    if use_attention and name.lower() == "densenet121":
        cnn_features = ECABlock(kernel_size=3, name="eca_attention")(cnn_features)
    cnn_pooled = layers.GlobalAveragePooling2D()(cnn_features)
    cnn_dense = layers.Dense(128, activation='relu', name='cnn_dense')(cnn_pooled)
    
    # Contour features branch (input will be provided separately during training)
    contour_input = layers.Input(shape=(contour_feature_dim,), name='contour_features')
    contour_dense1 = layers.Dense(32, activation='relu', name='contour_dense1')(contour_input)
    contour_dense2 = layers.Dense(64, activation='relu', name='contour_dense2')(contour_dense1)
    
    # Fusion: concatenate CNN and contour features
    fused = layers.Concatenate(name='fusion')([cnn_dense, contour_dense2])
    fused = layers.Dropout(dropout)(fused)
    fused = layers.Dense(128, activation='relu', name='fused_dense')(fused)
    fused = layers.Dropout(dropout * 0.5)(fused)
    outputs = layers.Dense(num_classes, activation="softmax", dtype="float32", name='predictions')(fused)
    
    # Model with two inputs: image and contour features
    model = models.Model([base.input, contour_input], outputs)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def build_hybrid_attention_contour_model(name, num_classes, base_trainable=False, dropout=0.4, 
                                        contour_feature_dim=4):
    """
    Build hybrid model: CNN + ECA Attention + Contour Fusion.
    This combines all improvements for maximum performance.
    
    Args:
        name: Base model name (e.g., "DenseNet121")
        num_classes: Number of output classes
        base_trainable: Whether base model is trainable
        dropout: Dropout rate
        contour_feature_dim: Dimension of contour features
    """
    return build_model_with_contour_fusion(name, num_classes, base_trainable, dropout, 
                                          contour_feature_dim, use_attention=True)
