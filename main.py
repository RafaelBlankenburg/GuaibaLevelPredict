from src.train import train_model
from src.predict import prever_altura

if __name__ == '__main__':
    train_model()

    exemplo = [
        [3.1, 2.8, 4.2, 3.9, 2.4, 2.6, 3.5, 3.2],
        [2.9, 2.6, 4.1, 3.7, 2.2, 2.4, 3.3, 3.0],
        [2.7, 2.4, 4.0, 3.5, 2.1, 2.3, 3.2, 2.9],
        [2.8, 2.5, 4.0, 3.6, 2.2, 2.5, 3.3, 3.0],
        [2.9, 2.6, 4.1, 3.7, 2.3, 2.6, 3.4, 3.1],
        [3.0, 2.7, 4.1, 3.8, 2.4, 2.6, 3.4, 3.2],
        [3.1, 2.8, 4.2, 3.9, 2.4, 2.6, 3.5, 3.2],
    ]

    resultado = prever_altura(exemplo)
    print(f'Altura prevista do rio: {resultado:.2f} metros')
