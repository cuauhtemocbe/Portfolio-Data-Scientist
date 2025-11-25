#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""Programa para ejecutar el juego de dados 'Zombie Dice'"""

__author__ = "Cuauhtémoc"
__version__ = "1.0"
__status__ = "Development"

from wasabi import msg
import time
import random
from typing import List, Tuple


class Dado:
    """Representa un dado con caras personalizadas.

    Args:
        caras (list[str]): Lista de valores posibles al lanzar el dado.
        color (str): Color que identifica el tipo de dado.
    """

    # Objeto fundamental: cada dado tiene caras distintas según su color.
    # Zombie Dice se basa en probabilidades distintas para cada tipo de dado.

    def __init__(self, caras: List[str], color: str = "blanco") -> None:
        self.caras = caras
        self.color = color
        self.resultado: str | None = None

    def lanzar(self) -> str:
        """Lanza el dado y devuelve el resultado obtenido.

        Returns:
            str: Cara seleccionada al azar.
        """
        # Elegimos una cara al azar del dado.
        self.resultado = random.sample(self.caras, 1)[0]
        return self.resultado


class Zombie:
    """Representa a un jugador zombie.

    Args:
        nombre (str): Nombre del jugador.
    """

    # Cada zombie acumula cerebros (puntos) pero si recibe 3 disparos
    # pierde todo lo ganado en la ronda.

    def __init__(self, nombre: str) -> None:
        self.nombre = nombre
        self.contador_cerebros = 0
        self.contador_disparos = 0
        self.contador_temporal_cerebros = 0

    def score(self, resultado: List[str]) -> None:
        """Actualiza los contadores temporales según el resultado de los dados.

        Args:
            resultado (list[str]): Resultados obtenidos en una tirada.
        """
        # Contamos cerebros y disparos obtenidos en la tirada actual.
        self.contador_disparos += resultado.count("disparo")
        self.contador_temporal_cerebros += resultado.count("cerebro")

        print("\nDisparos en la ronda", self.contador_disparos)
        print("Cerebros en la ronda", self.contador_temporal_cerebros)

        # Regla clave: 3 disparos → se pierde el turno y cerebros temporales.
        if self.contador_disparos >= 3:
            msg.fail("\n Acabo tu turno tuviste 3 disparos")
            self.restaurar_contadores_temporales()
            msg.good(f"> Total de cerebros: {self.contador_cerebros}")

    def terminar_turno(self) -> None:
        """Finaliza el turno del zombie y suma los cerebros temporales."""
        self.contador_cerebros += self.contador_temporal_cerebros
        self.restaurar_contadores_temporales()
        msg.good(f"{self.nombre} tienes {self.contador_cerebros} cerebros")

    def restaurar_contadores_temporales(self) -> None:
        """Reinicia los contadores temporales de disparos y cerebros."""
        self.contador_disparos = 0
        self.contador_temporal_cerebros = 0


class BolsaDados:
    """Gestiona los dados disponibles y permite robar y lanzar.

    Contiene la cantidad de dados según color y reconstruye la bolsa
    conforme se extraen.
    """

    def __init__(self) -> None:
        self.contador_dado_verde = 6
        self.contador_dado_amarillo = 4
        self.contador_dado_rojo = 3

        self.bolsa_dados: List[Dado] | None = None
        self.mano: List[Dado] | None = None
        self.colores: List[str] | None = None
        self.tirada_actual: List[str] | None = None
        self.crear_bolsa()

    def crear_bolsa(self) -> None:
        """Crea una bolsa con dados verdes, amarillos y rojos."""
        caras_verde = ["cerebro", "cerebro", "cerebro", "huellas",
                       "huellas", "disparo"]
        dado_verde = Dado(caras=caras_verde, color="verde")

        caras_amarillo = ["cerebro", "cerebro", "disparo", "huellas",
                          "huellas", "disparo"]
        dado_amarillo = Dado(caras=caras_amarillo, color="amarillo")

        caras_rojo = ["cerebro", "huellas", "huellas", "disparo",
                      "disparo", "disparo"]
        dado_rojo = Dado(caras=caras_rojo, color="rojo")

        # Se regeneran las instancias según los contadores actuales.
        # Esto simula que ciertos dados ya no están disponibles.
        self.bolsa_dados = (
            [dado_verde] * self.contador_dado_verde +
            [dado_amarillo] * self.contador_dado_amarillo +
            [dado_rojo] * self.contador_dado_rojo
        )

    def robar(self) -> None:
        """Extrae tres dados al azar y actualiza la bolsa."""
        # Si no hay suficientes dados, robamos lo que quede.
        if len(self.bolsa_dados) < 3:
            print("Ya no hay suficientes dados para robar")
            self.mano = self.bolsa_dados
        else:
            # Robar 3 dados: mecánica central del juego.
            self.mano = random.sample(self.bolsa_dados, 3)

        # Actualizamos colores robados para mostrarlos al usuario.
        self.colores = [dado.color for dado in self.mano]

        # Reducimos los contadores según los dados extraídos.
        # Así evitamos volver a sacar los mismos dados en la siguiente tirada.
        for dado in self.mano:
            if dado.color == "verde":
                self.contador_dado_verde -= 1
            elif dado.color == "amarillo":
                self.contador_dado_amarillo -= 1
            elif dado.color == "rojo":
                self.contador_dado_rojo -= 1

        # Reconstruimos la bolsa con las nuevas cantidades.
        self.crear_bolsa()

    def lanzar(self) -> None:
        """Roba y lanza los dados seleccionados."""
        # Lanzar implica robar primero y luego lanzar cada dado.
        self.robar()
        self.tirada_actual = [dado.lanzar() for dado in self.mano]


class TableroZombieDice:
    """Controla los turnos y jugadores del juego.

    Args:
        num_jugadores (int): Total de jugadores en la partida.
    """

    def __init__(self, num_jugadores: int) -> None:
        self.num_jugadores = num_jugadores
        self.zombies_list: List[Zombie] = []
        self.zombie_actual: Zombie | None = None
        self.index = 0
        self.creando_zombies()

    def creando_zombies(self) -> None:
        """Solicita el nombre de cada jugador y crea los objetos Zombie."""
        for i in range(1, self.num_jugadores + 1):
            nombre = input(f"Nombre del zombie {i}: ")
            zombie = Zombie(nombre)
            self.zombies_list.append(zombie)

        self.zombie_actual = self.zombies_list[self.index]

        print("\n == Nombre de los Zombies: == \n")
        for zombie in self.zombies_list:
            print(zombie.nombre, "\n")

    def siguiente_zombie(self) -> None:
        """Cambia al siguiente jugador en orden circular."""
        # Avanzamos al siguiente jugador de forma circular.
        self.index += 1
        if self.index >= len(self.zombies_list):
            self.index = 0
        self.zombie_actual = self.zombies_list[self.index]


def iniciar_juego() -> Tuple[TableroZombieDice, BolsaDados]:
    """Inicializa el tablero y la bolsa de dados.

    Returns:
        tuple: Tablero creado y nueva bolsa de dados.
    """
    # Punto crítico: se crean jugadores y la primera bolsa de dados.
    num_jugadores = int(input("Cuántos jugadores?: "))
    tablero = TableroZombieDice(num_jugadores)

    with msg.loading("Creando zombies 🧟..."):
        time.sleep(2)

    bolsa = BolsaDados()
    return tablero, bolsa


def iniciar_turno(tablero: TableroZombieDice) -> Tuple[TableroZombieDice, BolsaDados]:
    """Inicia el turno del siguiente jugador.

    Returns:
        tuple: Tablero actualizado y nueva bolsa.
    """
    tablero.siguiente_zombie()
    bolsa = BolsaDados()
    return tablero, bolsa


def lanzar(tablero: TableroZombieDice, bolsa: BolsaDados) -> Tuple:
    """Realiza una tirada de dados para el jugador actual."""
    # Acción principal del juego: lanzar dados y procesar resultados.
    bolsa.lanzar()

    print("\nDados robados: ", bolsa.colores)

    with msg.loading("Lanzando dados..."):
        time.sleep(0.5)

    print("\nResultado obtendo: ", bolsa.tirada_actual)

    # Actualizamos el estado del zombie según los resultados.
    tablero.zombie_actual.score(bolsa.tirada_actual)
    return tablero, bolsa


def pasar(tablero: TableroZombieDice) -> TableroZombieDice:
    """Finaliza el turno del jugador actual y actualiza el tablero."""
    # El jugador decide guardar sus cerebros acumulados
    # y pasa al siguiente turno.
    tablero.zombie_actual.terminar_turno()
    return tablero


def main() -> None:
    """Bucle principal del juego: controla flujo de turnos y acciones."""
    acciones_diponibles = ['lanzar', 'pasar', 'terminar']
    tablero = None

    msg.fail("Bienvenido a Zombie Dice 🧟🧠 \n\n")
    print("")

    text = """
    Eres un zombie con hambre de cerebros. Llega a 13 cerebros para
    ganar el juego, pero cuidado si recibes 3 disparos, porque perderás
    los cerebros obtenidos en la ronda.
    """
    msg.info(text)

    msg.warn("\nEscribe `terminar` en cualquier momento para terminar el juego")

    # El juego continúa hasta que el usuario escribe "terminar".
    while True:
        if not tablero:
            tablero, bolsa = iniciar_juego()

        # Se muestra siempre cuántos cerebros tiene el jugador actual.
        mensaje = (
            f"Turno del jugador: {tablero.zombie_actual.nombre}"
            f" y tiene {tablero.zombie_actual.contador_cerebros} cerebros"
        )
        msg.warn(mensaje)

        accion = input("\n Qué quieres hacer? [lanzar, pasar, terminar]: \n")

        # Validación básica de acciones.
        if accion not in acciones_diponibles:
            msg.warn("Acciones disponibles: ")
            print(acciones_diponibles)
            continue

        if accion == "lanzar":
            tablero, bolsa = lanzar(tablero, bolsa)

        if accion == "pasar":
            tablero = pasar(tablero)
            tablero, bolsa = iniciar_turno(tablero)

        if accion == "terminar":
            msg.info("Gracias por jugar Zombie Dice 🧟🧠")
            break


if __name__ == "__main__":
    main()
