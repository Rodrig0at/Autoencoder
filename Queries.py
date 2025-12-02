import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Definir la función de pérdida personalizada (necesaria para cargar el modelo U-Net)
def combined_loss(y_true, y_pred):
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    ssim_loss = 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
    return 0.7 * mse + 0.3 * ssim_loss

# Variables globales para modelos (se cargan bajo demanda)
autoencoder = None
encoder = None
decoder = None
autoencoder_unet = None
encoder_unet = None
decoder_unet = None

def cargar_modelos_clasicos():
    global autoencoder, encoder, decoder
    if autoencoder is None:
        print("[DEBUG] Cargando modelos clásicos...")
        try:
            autoencoder = load_model('autoencoder.h5')
            encoder = load_model('encoder.h5')
            decoder = load_model('decoder.h5')
            print("[DEBUG] Modelos clásicos cargados exitosamente")
        except Exception as e:
            print(f"[ERROR] No se pudieron cargar los modelos clásicos: {e}")
            raise

def cargar_modelos_unet():
    global autoencoder_unet, encoder_unet, decoder_unet
    if autoencoder_unet is None:
        print("[DEBUG] Cargando modelos U-Net...")
        try:
            autoencoder_unet = load_model('autoencoder_unet.h5', 
                                           custom_objects={'combined_loss': combined_loss},
                                           compile=False)
            encoder_unet = load_model('encoder_unet_.h5', compile=False)
            decoder_unet = load_model('decoder_unet_.h5', compile=False)
            print("[DEBUG] Modelos U-Net cargados exitosamente")
        except Exception as e:
            print(f"[ERROR] No se pudieron cargar los modelos U-Net: {e}")
            raise

# Función para cargar imagen
def cargar_imagen(ruta, unet=False):
    if unet:
        img = image.load_img(ruta, color_mode='rgb', target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        return img_array.reshape(1, 224, 224, 3)
    else:
        img = image.load_img(ruta, color_mode='grayscale', target_size=(28, 28))
        img_array = image.img_to_array(img) / 255.0
        return img_array.reshape(1, 28, 28, 1)

# Funciones de modelo

# Modelos clásicos
def reconstruir(img_array):
    return autoencoder.predict(img_array)

def codificar(img_array):
    return encoder.predict(img_array)

def decodificar(latente):
    return decoder.predict(latente)

# Modelos U-Net
def reconstruir_unet(img_array):
    return autoencoder_unet.predict(img_array)

def codificar_unet(img_array):
    return encoder_unet.predict(img_array)

def decodificar_unet(latente):
    return decoder_unet.predict(latente)

# Visualización
def mostrar(original, reconstruida, unet=False):
    print("[DEBUG] Mostrando imágenes...")
    plt.subplot(1, 2, 1)
    if unet:
        plt.imshow(original.squeeze())
    else:
        plt.imshow(original.squeeze(), cmap='gray')
    plt.title("Original")

    plt.subplot(1, 2, 2)
    if unet:
        plt.imshow(reconstruida.squeeze())
    else:
        plt.imshow(reconstruida.squeeze(), cmap='gray')
    plt.title("Reconstruida")

    plt.show()
    # Guardar la imagen reconstruida en disco para depuración
    from PIL import Image
    arr = (reconstruida.squeeze() * 255).astype('uint8')
    if unet:
        img = Image.fromarray(arr)
    else:
        img = Image.fromarray(arr, mode='L')
    img.save('reconstruida_debug.png')
    print("[DEBUG] Imagen reconstruida guardada como 'reconstruida_debug.png'")

# Main con argparse

def main():
    parser = argparse.ArgumentParser(description="Procesa una imagen con el autoencoder clásico o U-Net")
    parser.add_argument('--ruta', type=str, required=True, help='Ruta a la imagen')
    parser.add_argument('--modelo', type=str, choices=['clasico', 'unet'], default='unet', help='Modelo a usar: clasico o unet')
    parser.add_argument('--modo', type=str, choices=['autoencoder', 'encoder', 'decoder'], default='autoencoder', help='Modo de operación')
    parser.add_argument('--mostrar', action='store_true', help='Mostrar imagen original y reconstruida')
    args = parser.parse_args()

    print("[DEBUG] Iniciando script...")
    print(f"[DEBUG] Argumentos: ruta={args.ruta}, modelo={args.modelo}, modo={args.modo}, mostrar={args.mostrar}")
    
    usar_unet = args.modelo == 'unet'
    
    # Cargar solo el modelo necesario
    try:
        if usar_unet:
            cargar_modelos_unet()
        else:
            cargar_modelos_clasicos()
    except Exception as e:
        print(f"[ERROR] No se pudo cargar el modelo. Saliendo...")
        return
    
    print(f"[DEBUG] Cargando imagen: {args.ruta} (unet={usar_unet})")
    try:
        img_array = cargar_imagen(args.ruta, unet=usar_unet)
        print(f"[DEBUG] Imagen cargada. Shape: {img_array.shape}")
    except Exception as e:
        print(f"[ERROR] No se pudo cargar la imagen: {e}")
        return

    print(f"[DEBUG] Ejecutando modelo: {'U-Net' if usar_unet else 'Clásico'} en modo {args.modo}")
    try:
        if usar_unet:
            if args.modo == 'autoencoder':
                print("[DEBUG] Llamando a reconstruir_unet...")
                salida = reconstruir_unet(img_array)
            elif args.modo == 'encoder':
                print("[DEBUG] Llamando a codificar_unet...")
                salida = codificar_unet(img_array)
            elif args.modo == 'decoder':
                print("[DEBUG] Llamando a decodificar_unet...")
                salida = decodificar_unet(img_array)
        else:
            if args.modo == 'autoencoder':
                print("[DEBUG] Llamando a reconstruir...")
                salida = reconstruir(img_array)
            elif args.modo == 'encoder':
                print("[DEBUG] Llamando a codificar...")
                salida = codificar(img_array)
            elif args.modo == 'decoder':
                print("[DEBUG] Llamando a decodificar...")
                salida = decodificar(img_array)
        print("[DEBUG] Predicción completada")
    except Exception as e:
        print(f"[ERROR] Error durante la predicción: {e}")
        import traceback
        traceback.print_exc()
        return

    print(f"Modelo: {'U-Net' if usar_unet else 'Clásico'}")
    print(f"Modo: {args.modo}")
    print("Shape de salida:", salida.shape)

    if args.mostrar and args.modo == 'autoencoder':
        mostrar(img_array, salida, unet=usar_unet)
    print("[DEBUG] Script finalizado.")

if __name__ == "__main__":
    main()