#!/usr/bin/python3         
#-*- coding: utf-8 -*-     

"""Programa para ejecutar el juego de dados 'Zombie Dice'"""      

__author__ = "Cuauhtémoc"
__version__ = "1.0"
__status__ = "Development"

from wasabi import msg
import time
import random

class Dado:
  def __init__(self, caras: list, color: str = 'blanco'):

    self.caras = caras
    self.color = color
    self.resultado = None

  def lanzar(self):
    self.resultado = random.sample(self.caras, 1)[0]

    return self.resultado

class Zombie:
  def __init__(self, nombre):
     self.nombre = nombre
     self.contador_cerebros = 0
     self.contador_disparos = 0
     self.contador_temporal_cerebros = 0

  def score(self, resultado):
     self.contador_disparos += resultado.count('disparo')
     self.contador_temporal_cerebros += resultado.count('cerebro')

     print('\nDisparos en la ronda', self.contador_disparos)
     print('Cerebros en la ronda', self.contador_temporal_cerebros)

     if self.contador_disparos >= 3:
       msg.fail("\n Acabo tu turno tuviste 3 disparos")
       self.restaurar_contadores_temporales()
       msg.good(f"> Total de cerebros: {self.contador_cerebros}")


  def terminar_turno(self):
    self.contador_cerebros += self.contador_temporal_cerebros
    self.restaurar_contadores_temporales()
    msg.good(f"{self.nombre} tienes {self.contador_cerebros} cerebros")

  def restaurar_contadores_temporales(self):
    # Se reinicia los contadores para el próximo turno
    self.contador_disparos = 0
    self.contador_temporal_cerebros = 0

class BolsaDados:
  def __init__(self):
    self.contador_dado_verde = 6
    self.contador_dado_amarillo = 4
    self.contador_dado_rojo = 3

    self.bolsa_dados = None
    self.mano = None # objetos dados en la mano
    self.colores = None
    self.tirada_actual = None # resultados obtenidos
    self.crear_bolsa()

  def crear_bolsa(self):

    caras_verde = ['cerebro', 'cerebro', 'cerebro', 'huellas',
                   'huellas', 'disparo']
    dado_verde = Dado(caras=caras_verde, color='verde')

    caras_amarillo = ['cerebro', 'cerebro', 'disparo', 'huellas',
                      'huellas', 'disparo']
    dado_amarillo = Dado(caras=caras_amarillo, color='amarillo')

    caras_rojo = ['cerebro', 'huellas', 'huellas', 'disparo', 'disparo',
                  'disparo']
    dado_rojo = Dado(caras=caras_rojo, color='rojo')

    # Siempre iniciando con una bolsa vacía
    self.bolsa_dados = []

    for i in range(self.contador_dado_verde):
      self.bolsa_dados.append(dado_verde)

    for i in range(self.contador_dado_amarillo):
      self.bolsa_dados.append(dado_amarillo)

    for i in range(self.contador_dado_rojo):
      self.bolsa_dados.append(dado_rojo)

  def robar(self):
    # Creando nuestra mano inicial
    if len(self.bolsa_dados) < 3:
      print('Ya no hay suficientes dados para robar')
      self.mano = self.bolsa_dados
    else:
      self.mano = random.sample(self.bolsa_dados, 3)
    # Colores en la mano
    self.colores = [dado.color for dado in self.mano]

    # Conteo de colores para actualizar
    # nuestra bolsa
    for dado in self.mano:
      if dado.color == 'verde':
        self.contador_dado_verde -= 1

      elif dado.color == 'amarillo':
        self.contador_dado_amarillo -= 1

      elif dado.color == 'rojo':
        self.contador_dado_rojo -= 1

    # Generamos nuevamente nuestra bolsa de dados
    # con el nuevo número de dados disponibles
    self.crear_bolsa()


  def lanzar(self):
    self.robar()
    self.tirada_actual = [dado.lanzar() for dado in self.mano]

class TableroZombieDice:

  def __init__(self, num_jugadores):
    self.num_jugadores = num_jugadores
    self.zombies_list = []
    self.zombie_actual = None
    self.index = 0 # id del jugado actual
    self.creando_zombies()

  def creando_zombies(self):
    for i in range(1, self.num_jugadores + 1):
      nombre = input(f'Nombre del zombie {i}: ')
      zombie = Zombie(nombre)
      self.zombies_list.append(zombie)

    self.zombie_actual = self.zombies_list[self.index]

    print(f'\n == Nombre de los Zombies: == \n')

    for zombie in self.zombies_list:
      print(zombie.nombre, '\n')

  def siguiente_zombie(self):

    self.index += 1

    if self.index  > len(self.zombies_list) - 1:
      self.index  = 0

    self.zombie_actual = self.zombies_list[self.index]

def iniciar_juego():
  num_jugadores = int(input('Cuántos jugadores?: '))
  tablero = TableroZombieDice(num_jugadores)

  with msg.loading("Creando zombies 🧟..."):
    time.sleep(2)

  bolsa = BolsaDados()

  return tablero, bolsa

def iniciar_turno(tablero):
  tablero.siguiente_zombie()
  bolsa = BolsaDados()

  return tablero, bolsa

def lanzar(tablero, bolsa):
  bolsa.lanzar()

  print('\nDados robados: ', bolsa.colores)


  with msg.loading("Lanzando dados..."):
    time.sleep(0.5)

  print('\nResultado obtendo: ', bolsa.tirada_actual)

  tablero.zombie_actual.score(bolsa.tirada_actual)

  return tablero, bolsa

def pasar(tablero):
  tablero.zombie_actual.terminar_turno()

  return tablero

# Finalmente solo hay que ponder todo dentro de un main
# para poder ejecutarlo

def main():
  tablero = None
  acciones_diponibles = ['lanzar', 'pasar', 'terminar']

  msg.fail("Bienvenido a Zombie Dice 🧟🧠 \n\n")
  print('')

  text = """
  Eres un zombie con hambre de cerebros. LLega a 13 cerebros para
  ganar el juego, pero cuidado si recibes 3 disparos, por que perderas
  los cerebros obtenidos en la ronda
  """
  msg.info(text)

  msg.warn("\nEscribe `terminar` en cuallquier momento para terminar el juego")

  while True:

    if not tablero:
      tablero, bolsa = iniciar_juego()

    mensaje = f"Turno del jugador: {tablero.zombie_actual.nombre}" \
        f" y tiene {tablero.zombie_actual.contador_cerebros} cerebros"

    msg.warn(mensaje)

    accion = input('\n Qué quieres hacer? [lanzar, pasar, terminar]: \n')

    if accion not in acciones_diponibles:
      msg.warn('Acciones disponibles: ')
      print(acciones_diponibles)

    if accion == 'lanzar':
      tirada_actual = lanzar(tablero, bolsa)

    if accion == 'pasar':
      tablero = pasar(tablero)
      tablero, bolsa = iniciar_turno(tablero)

    if accion == 'terminar':
      msg.info('Gracias por jugar Zombie Dice 🧟🧠')
      break

if __name__ == "__main__":
    main()
