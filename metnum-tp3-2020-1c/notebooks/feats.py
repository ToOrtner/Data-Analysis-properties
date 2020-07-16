import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None
ciudades = np.array(["San Pedro Garza García", "Colima", "Mérida", "San Nicolás de los Garza", "Saltillo", "Mazatlán", "Apodaca", "Chihuahua",
                     "Aguascalientes", "Mexicali", "Querétaro", "Campeche", "Guadalupe", "Matamoros", "Nuevo Laredo", "Venustiano Carranza", "Torreón",
                     "León", "Culiacán", "Hermosillo", "Monterrey", "Reynosa", "Benito Juárez", "Zapopan", "Zacatecas", "La Paz", "Manzanillo",
                     "Iztapalapa", "Gómez Palacio", "Guanajuato", "Cuajimalpa de Morelos", "Veracruz", "Lázaro Cárdenas", "Miguel Hidalgo",
                     "Guadalajara", "Azcapotzalco", "Tlaquepaque", "Carmen", "Cuauhtémoc", "Gustavo A. Madero", "Tepic", "Coyoacán",
                     "Iztacalco", "Benito Juárez", "Pachuca", "Milpa Alta", "Durango", "Juárez", "Tlaxcala", "Toluca", "Tlalpan", "Alvaro Obregón",
                     "Chimalhuacán", "Nezahualcóyotl", "Tlalnepantla de Baz", "Chilpancingo de los Bravo", "Xalapa", "Cajeme", "Acapulco de Juárez", "Xochimilco",
                     "San Luis Potosí", "La Magdalena Contreras", "Villahermosa", "Tijuana", "Morelia", "Oaxaca de Juárez", "Cuernavaca", "Tapachula",
                     "Naucalpan de Juárez", "Tláhuac", "Tuxtla Gutiérrez", "Puebla", "Victoria", "Chetumal", "Tehuacán", "Ecatepec de Morelos"])


def newfeats(df):

    #valor predeterminado para las ciudades que no esten en este ranking
    df['mejorciudad'] = 100
    #valor que llevará la ciudad según su posición en el ranking
    posicion = 1
    for c in ciudades:
        df.loc[df['ciudad'] == c, 'mejorciudad'] = posicion
        posicion += 1


    #tropico de cancer = a latitud  23,43
    df = df.dropna(subset=['lat', 'piscina'])

    df['calurosa'] = (df['lat'] < 23.43) & (df['piscina'] > 0)

    #escuelas
    df = df.dropna(subset=['habitaciones', 'escuelascercanas'])

    df['parachicos'] = (df['habitaciones'] > 2) & (df['escuelascercanas'] > 0)

    df['urbana'] = (df['escuelascercanas'] > 0) & (df['centroscomercialescercanos'] > 0)

    return df

