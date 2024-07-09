# Aspectos cruciales
1) El tipo de problema de optimizacion (maximizar o minimizar)
2) El tipo de objeto de python o plantilla que va a contener el individuo 
3) el objeto caja de herramientas o toobox que contendrá, mediante registro,
un conjunto de funciones utilizadas por el algoritmo durante su ejecución. 
    Entre los tipos de funciones que se registran, destacan las siguientes: 
        a. las funciones para crear los individuos de forma aleatoria
        b. la función para crear la población. 
        c. los operadores genéticos (selección, cruce y mutación)
        d. la función objetivo. 

---------- INTRODUCCIÓN -------------

__Creación del problema__ 

El método create crea una nueva clase llamada FitnessMax que hereda de base.Fitnesss y que tiene un atributo que se denomina weights. 

Esta linea de código merece más detalles. El método tiene los siguientes argumentos: 

- name: nombre de la clase que se crea. 
- base: Clase de la uqe hereda.
- atribute: Uno o más atributos que se quieran añadir a la clase cuando se cree . 

El objeto base contiene los atributos encargados de almacenar el fitness o desempeño de un individuo. En concreto, contiene los siguientes atributos: 

a. values: Es una tupla que contiene los valores de fitness de cada uno de los objetivos de nuestro problema. Pueden tratarse de varios objetivos, llamados multiobjetivo, pues contendrá la calidad de cada individuo en cada uno de los objetivos de nyuestro problema de 
optimización.

b. dominates: Devuelve verdadero (True) si una solución es estrictamente peor que otra. Este atributo se utilizará en los algoritmos genéticos con múltiples objetivos. 

c. valid: Indica si el fitness de un individuo es válido. Este atributo se utiliza para saber el número de individuos que se tienen que evaluar en cada iteración del algoritmo genético. En general, si un individuo tiene el atributo values vacío, el atributo valid será False.

Una forma de comprender como opera la herencia, es traduciendolo a clase: 

```
from deap import base 

class MaxFitness(base.Fitness): 
    weights = (1.0,)
```

En si, tenemos que crear dos cosas en creator.create: 
1. El tipo de problema (maximizar1,0 o minimizar -1,0)
2. El número de objetivos que tiene nuestro problema (uno o varios, según unos o menos unos contenga la tupla del atributo weights).

__Creación del Individuo__

En esta línea de código estamos creando una clase que se denomina Individual, que hereda de la clase lista (por lo tanto, tiene todos los métodos de una lista) y que contiene el atributo fitness, el cual ha sido inicializado con el objeto fitnessMax creado en la anterior linea. Es decir, el indivduiop será una lista que contiene un atributo fitness que almacenará la calidad o desempeño de este. 

Sería como hacer 
```
from deap import base 

class MaxFitness(base.Fitness): 
    weights = (1.0,)

class Individual(list): 
    fitness = MaxFitness
```

Definir los individuos como una lista nos permite poder acceder a cada uno de los genes mediante la posición que ocupa. Cada posición de la secuencia es una variable distint. 

**Ya hemos definido el tipo de problema y el tipo de inviduio que vamos a utilizar. Estos dos pasos se van a dar siempre y, en la mayoría de los casos, ambas líneas de código se repetiran con pequeñas modificaciones dependiendo el número de objetivos y del tipo de objeto que almacene los individuos.**

El tamaño de la lista se define cuando se crean los individuos de la población inicial. 

__Crear individuos aleatorios y población inicial__
Al definir funciones con la base.Toolbox, obtenemos las funciones que nos permiten crear individuuos. 

```
toolbox = base.Toolbox()
```

Este objeto permite registrar funciones que se utilizarán durante la operación del algoritmo genético. El registro de funciones se realiza mediante el método register de la clase base.Toolbox. 

El método register tiene los siguientes atributos: 

1. __alias__: El nombre con el que registramos la función en la caja de herramientas
2. __function__: la función que estamos registrando en la caja de herramientas. 

En primer lugar, se registrarán las funciones que nos permiten crear individuios aleatorios. Para eso, necesitamos desarrollar una función que nos permita generar un valor aleatorio para cada variable indpependiente (cada gen del cromosoma), esto es, cada una de las posiciones de la lista. 

Conviene que dicho valor esté comprendido entre cada uno de los límites de uestras variables 

```
toolbox.register("attr_uniform", random.uniform, -100, 100)
```

El equivalente en Python sería: 
```
import random 

def funcion(a=-100, b=100):
return random.uniform(a, b)

class Toolbox(object): 
attr_uniform = function

```

Con el fin de crear un individuo completo, se registra la funcion Individual. Que llamará a la función tools.initrepeat

tools.initrepeat tiene como parámetros: 

1. __container__: El tipo de dato dónde se almacenará el resultado del argumento func. 
2. __func__: Función a la que se llamará n veces. 
3. __n__ : Número de veces que se llamará a la función func. 

El método initrepeat nos permite jecutar varias veces la función registrada attr_uniform y almacenar el resultado en el individuo que queremos crear. Como resultado, se crea un individuo aleatorio. 

Así, cada vez que se ejecute toolbox.individual() se creará un individuo aleatorio con su atributo fitness, que almacenará la calidad del mismo. 
Dicho atributo debe estar creado junto con el individuo y, además, debe estar vacío, ya que el individuo todavía no ha sido evaluado. 

Para acceder al fitness usamos 
```
individuo.fitness.values
```

__El procedimiento para crear la población inicial es análogo. __ La diferencia sustancial está en que el container será una lista y la función a llamar será toolbox.individual. 

Con respecto al tamaño de las poblaciónes, no existe un tamaño óptimo de población para los problemas de optimización, pero debe estar en proporción al numero de variables independientes que tengamos. 

__Cuanto mayor será el numero de variables independientes, mayor debe ser el tamaño de la población inicial__
Se puede establecer que el tamaño de la población inicial sea un orden dee magnitud mayor al número de variables independientes. 

Al no obtener resultados satisfactorios, se puede aumentar el tamaño de la poblacion. 


**Como resumen**: 
1. Los problemass solo pueden ser de dos tipos (maximizar o minimizar). Los multiobjetivos se verán más adelante. 
2. Los individuios serán listas en la mayoría de los casos. Por lo tanto, la línea de creación nos valdrá para la mayoría de casos: 

```
creator.create("Individual", list, fitness=creator.FitnessMax)
```

3. Necesitamos una función para generar cada uno de los genes de nuestro individuo. Esto sí será diferente para cada problema. Aunque en la mayoría de los problemas con variables continuas la función random-uniform nos puede valer, en el resto de casos simplementetendremos uqe cambiar los límites.
4. Una vez que tenemos la función para generar los genes de nuestro individuo, el registro de funciones para crear individuos aleatorios y la población inicial serán casi siempre los mismos. Lo único que podemos cambiar es el tamaño de los indivduos y la población. 

**1.4 Funciópn objetivo y operadores genéticos**

Se procederá con el registro de la función objetivo y los operadores genéticos 
```
def funcion_objetivo(x): 
    for i in range(len(x)):
        if x[i] > 100 or x[i] < -100: 
            return -1, 
    res = math.sqrt(x[0]**2 + x[1]**2)
    return res, 

toolbox.register("evaluate", funcion_objetivo)
toolbox.register("mate", tools.cxOnePoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma = 5, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)
```

__Función objetivo__
La parte más particular del problema de optimización. Podemos dividarlas en dos tipos: 
1. Funciones objetivo que están codificadas en Python 
2. Funciones objetivo que son el resultado de un programa o software externo. 

En el primer caso, debemos codificar la función objetivo de nuestro problema como una función de Python. El módulo nativo __mat__ y las librerías __numpy y scipy__ pueden ser útiles, ya que contienen una gran cantidad de funciones matemáticas disponibles.

En el segundo caso, nuestro __script___ de Python llamará a un programa externo para obtener el desempeño del individuo. Este segundo caso nos permite utilizar modelos más complejos incluidos en software específico. 

Se destacará que existen funciones en Python para esto pero no se profundizará. 


Esta sería la parte dónde usamos Python apra la función 

```
def funcion_objetivo(x): 
    for i in range(len(x)):
        if x[i] > 100 or x[i] < -100: 
            return -1, 
    res = math.sqrt(x[0]**2 + x[1]**2)
    return res, 
```

Utilizamos math para calcular la raiz cuadrada. 

Es importante destacar el hecho de que si una solución no cumple las restricciones, debe ser descartada. Si una de las variables independientes toma valores fuera del dominio de la función, la función objetivo devolverá un -! 

Esto se conoce como ***aplicar la pena de muerte***. 

La pena de muerte hace que un individuo no participe de las funciones de ***cruce y mutación***

__La pena de muerte es un mecanismo por el cual se inhabilita a un individuo de una determinada población a participar en las operaciones genéticas. La pena de muerte consiste en asignar un desempeño muy malo al individuo.__

Para registrar la función de fitness, se procede así 

```
toolbox.register("evaluate", funcion_objetivo)
```

Se puede evaluar al individuo pasandolo como parámetro a la función o utilizando __toolbox.evaluate(individuo)__ ya que está registrada. 

Un detalle importante que está relaciónado con el módulo deap es que ***la función de fitness devuelve una tupla, con independencia del número de objetivos del problema***

__El caso con un único objetivo no es más que un caso particular del problema genérico multiobjetivo en deap.__

Siempre debe devolver una tupla, aunque uno de los componentes este vaco. 

Es de importancia destacar la relevancia de codificar de manera eficiente la función objetivo porque se ejecutará unaa gran cantidad de veces, por lo que todo ahorro supondrá una gran ventaja. 

Siempre que podamos, deberemos evitar bucles o condiciones que puedan dejar colgado el algoritmo. 

***1.5 Operadores genéticos***

__Las operaciones genéticas son aquellos mecanismos que nos permiten generar nuevos individuos, también llamados ***descendencia*** u ***offspring***, en base a la población actual__

Las operaciones genéticas son de tres tipos: 
1. Seleccion (Selection): Proceso en el cual se seleccionan los indivudios que participarán en als operaciones de cruce y mutación. Es un procedimiento elitista, un individuo quedará seleccionado si su fitness es más adecuado al problema de optimización.
2. Cruce (Mate): Operación probabilística que permite que dos individuos seleccionados crucen o intercambien su información genética para crear dos nuevos individuos. Es importante indicar de nuev, que la operación de cruce es probabilística; esto quiere decir qeu, aunque dos individuos sean seleccionados, puede qeu no sean modificados. La probabilidad de cruce es un hiperparámetro de los algoritmos genéticos que tendremos que definir. No existe un valor óptimo universal por lo que deberá ser ajustado a cada problema.
3. Mutación (Mutation): Es una operación probabilistica que permite que un individuo seleccionado modifique su información genética para crear un nuevo individuo. Al igual que el cruce, la mutación es una operación probabilística cuyo resultado depende de la probabilidad de mutación, la cual también debemos definir nosotros como otro hiperparámetro. También ajustable a cada problema. 

En un algoritmo genético clásico o canónico, primero se realiza la selección de individuos. Se cruzan, en caso de que la probabilidadd sea favoralble, y después se mutan., 

Como ambas operaciones son probabilísticas, se puede dar el caso de que un individuo que se ha seleccionado no sea modificado debido a que ninguna de las probabilidades le sea favorable. 

En estos casos, **pasaría a la siguiente generación sin ningún tipo de modificación** 

El ajuste de las probabilidades de cruce y mutacióne es sumamente importante para el funcionamiento adecuado de un algoritmo genético. 

***1.5.2 Operadores en deap***
Deap cuenta con variedad de algoritmos de selección, cruce y mutación. 

La aplicación de cada una de ellas dependerá del problema, ya uqe algunas operaciones son adecuadas para problemas con variables continuas y otras con variables discretas. S debe utilizar la documentación. 

__Inicialización__
initRepeat()
initIterate()
initCycle()

__Crossover__
cxOnePoint()
cxTwoPoint()
csUniform()
cxpartialyMatched()
cxUniformPartialyMatched()
cxOrdered()
cxBlend()
cxSimulatedBinary()
cxSimulatedBinarybounded()
cxMessyOpePoint()

__Mutación__
mutGaussian()
mutShbuffleIndexes()
mutBitFlip()

__Selección__
selTournament()
selRoulette()
selNSGA2()
selNSGA3()
selSPEA2()
selRandom()
selBest()
selWorst()
selTournamentDCD()
selDoubleTournament()
selDoubleTournament()
selStochasticUniversalSampling()
sellexiCase()
selEpsilonLexicase()
selAutomaticEpsilonLexicase()

__Migración__
migRing()

El registro de operadores a utilizar será el siguiente: 

1. Definiremos el mecanismo que utilizaremos para realizar el mate entre individuos. En este caso, será cxOnePoint o cruce de un punto. 

```
toolbox.register("mate", tools.cxOnePoint)
```

__Operador cxOnePoint__
***Este operador de cruce selecciona de manera aleatoria un punto (una posición) dentro de los individuos, ind1 e ind2, para hacer un intercambio entre la información genética de dos individuos (padres) seleccionados


-01001010 
-11000100

*01001 100
*11000 010

2. Para la mutación, se ha utilizado el operador mutGaussian con una media de cero y una desviación típica de 5. Estos valores son solo de ejemplo 


```
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma = 5, indpb=0.1)
```

__Operador mutGaussian__
***Este operador de mutación consiste en mutar, con una probabilidad indpb cada uno de los genes de lindividuo individual, sumando a cada uno de ellos una cierta cantidad aleatoria. Esta cantidad se genera según una función de probabilidad gaussiana (o normal), de media mu y desviación típica sigma***


Es importante elegir adecuadamente el parámetro indpb que define la probabilidad de mutación de cada gen. 

Las probabilidades se definirán más adelante. 

En el caso de la mutación, se deben definir dos probabilidades: 
a. La probabilidad de mutar un individuo. 
b. La probabilidad de mutar cada uno de los genes del individuo(indpb)

En el ejemplo se ha elegido una probabilidad de 0.1. En general, debe ser bajo para que no modifique en exceos al individuo. 



3. Aplicación del proceso de selección mediante selTournament, que nos permitirá realizar una selección mediante toreo. En este caso, el tamaño será de 3. Este tamaño suele funcionar relativamente bien para la mayoría de los casos

```
toolbox.register("select", tools.selTournament, tournsize=3)
```

__Operador selTournament__
La selección mediante torneo funciona de la siguiente forma: 
a. Se selecciona de manera aleatoria un número tournsize de individuos de la población individuals. 
b. En cada selección existen reemplazos por lo que un individuo puede ser seleccionado varias veces en un mismo torneo. 
c. Los individuos seleccionados compiten entre ellos para ganar el torneo utilizando el desempeño obtenido por cada de uno al evaluar la función de fitness. 
d. El individuo con mejor desempeño es el que ganará el torneo, y será quien participe en las operaciones genéticas. 
e. Este proceso se repite hasta seleccionar un total de k individuos. 


El algoritmo realiza tnatos torneos como individuos tiene la pbolación ya que primero se cruzarán y luego mutaran. 

***Cuando la población crece mucho, se deben utilizar tamaños más altos para hacer más rápida la convergencia del algoritmo*** 
Es una selección muy elitista y hace que el algoritmo converja a mayorvelocidad si lo comparamos con otros algoritmos de seleccion como la selección mediante ruleta.

***Los alias del registro de los operadores genéticos deben ser: mate para el cruce, mutate para la mutación y selection para la selección. No se debe modificar si se va a utilizar el algoritmo genético como caja negra. También ocurre lo mismo con la función objetivo, la cual se debe registrar con el alias evaluate. Todo esto mediante el método register de toolbox.***

En el caso del cruce, el objetivo es encontrar bloques dentro de la cadena cromosómica que den origen a buenos resultados de la función de evaluación. Estos bloques serán intercambiados con mayor probabilidad a otros individuos. Por lo tanto, tenderán a parecerse los unos a los otros a lo largo de las generaciones. Estará limitado por los valores máximos y mínimos de los genes en la población inicial. 

El operador de mutación nos permite ampliar el área de las posibles soluciones de la población inicial (y cualquier otra generación), __incrementando los valores máximos y mínimos de las variables independientes__

Por último, la operación de selección nos permite aplicar un componente elitista al algoritmo, de manera que auqellos individuos que mejor se adapten serán los que con mayor probabilidad intercambien sus genes o muten. 

**1.6 Algoritmo genético como caja negra**
Realizaremos la función main que ejecutara el algoritmo. 

```
def main:
    random.seed(42)
    CXPB, MUTB, NGEN = 0.5, 0.2, 20
    pop = toolbox.population()
    hof = pop.HallOfFame(1)
    
```

__1.6.1 Configuración algoritmo genético__
La primera sentencia define la semilla del generador de números aleatorios. 

```
    random.seed(42)
```

Luego, se definen tres parámetros importantes para el funcionamiento del algoritmo: 
```
    CXPB, MUTB, NGEN = 0.5, 0.2, 20
```


- La probabilidad del cruce (CXPB)
- La probabilidad de mutación (MUTPB)
- El número de generaciónes (NGEN)

En cuanto a las probabilidades: 
- 0.5 para el cruce (50%).
- 0.2 para la de mutación (20%). 
- 20 para el número de generaciones.

Luego comenzamos con la generación de la población inicial metiante el método __population__ del objeto __toolbox__

```
    pop = toolbox.population()
```

Se define un objeto hof de tipo HallOfFame para almacenar al mejor individuo encontrado a lo largo de las generaciones 

```
    hof = tools.HallOfFame(1)
```
El método HOF recibe dos parámetros: 
- maxsize: Número de individuos a almacenar. 
- similar: Una función para comparar si dos individuos son iguales. Si no se pone nada, utilizará por defecto el kétodo operator.eq del módulo operator. 

En este caso, maxsize es 1. Este es el mecanismo que implementa deap para no perder al mejor indivudo a lo largo de la evolución del algoritmo. 
Con el algoritmo eaSimple se puede dar el caso de que la mejor solución se pierda debido a los operadores geneticos. Es por ello que el hof es importante. 

La clase se encuentar definida en el submódulo tools. En cada generación del algoritmo genético, el objeto hof es actualizado mediante el método update de manera transparente para nosotros. El método update recibe como entrada la población actual y adctualiza el contenido del objeto hof. 

En este caso, no se ha utilizado elitismo para que los mejores individuos avancen directamente. 

Se define un objeto para generar las estadísticas de la población a lo largo de las generaciones del algoritmo. Este objeto es de tipo __Statistics__ y se encuentra definido en el submódulo tools. 

```
stats = tools.Statistics(lambda ind: ind.fitness.values)
```

Al crear el objetom, se debe indicar sobre qué atributo de los individuyos se van a generar las estadísticas. En este caso, se generan sobre el __fitness__ de los individuos mediante calculos de la media, la desviación típica, el mínimo y el máximo.

```
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)
```

Dichos cálculos se realizaran de manera transparente. El objeto stats tiene un método denominado __compile__ que recibe como entrada la población que permite generar las estadísticas. A dicho método se lo llama internamente en cada generación del algoritmo. 

```
La librería deap doispone de varias implementaciones de algoritmos genéticos listos para ser utilizados de manera sencilla como cajas negras o black boxes.
```

Se encuentran disponibles en el submódulo __algorithms__

```
***Algoritmo genético: EaSimple***

Es una de la sversiones simplificadas que hay disponibles en deap.

Sus pasos son los siguientes: 
    1. Evaluación de la población inicial.
    2. Mientras no se alcanza el límite de generaciones: 
        a - Selección de individuos para participar de las operaciones genéticas.
        b - Creación de la siguiente generación mediante cruce y/o mutación de los individuos seleccionados. 
        c - Evaluación de la generación creada.
La librería lo incorpora mediante __algorithm.eaSimple__

Los parámetrosa indicar son: 
- __population__ : Población inicial del algoritmo. 
- __toolbox__ : Objeto caja de herramientas con las siguientes funciones registradas: función objetivo, mecanismos de selección, cruce y mutación.
- __expb__ : Probabilidad de cruce
- __mutb__ : Probabilidad de mutación en tanto por uno. 
- __ngen__ : Número de generaciones del algoritmo.
- __stats__ : Objeto estadístico con las funciones registradas. 
- __halloffame__ : Objeto de tipo HallOfFame para almacenar la mejor solución encontrada.
- __verbose__ : Atributo que indica si se deben mostrar las estadísticas en cada iteración o no. 

Como resultado, devuelve: 
- Población final del algoritmo.
- Objeto __logbook__ con todo el registro de evolución.
```

**Algoritmo genético completo**
```
def main():
    random.seed(42)
    CXPB, MUTPB, NGEN = 0.5, 0.3, 20
    
    pop = toolbox.population()
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    logbook = tools.Logbook()
    pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb = CXPB, mutpb= MUTPB, ngen=NGEN,stats=stats, halloffame=hof, verbose=True)
    
    return hof, logbook

if __name__ == '__main__':
    best, log = main() 
    print("Mejor fitness: %f" %best[0].fitness.values)
    print("Mejor individuo: %s" %best[0])
```

La función main devolverá el mejor individuo así como la población final. 
```
gen     nevals  avg     std     min     max    
0       30      77.9162 24.544  26.4289 116.497
1       21      98.2664 15.5883 58.7221 123.194
2       21      110.245 15.2091 69.9853 134.549
3       18      122.618 6.15311 112.506 134.549
4       20      128.053 6.19413 112.506 134.549
5       27      132.693 3.40979 123.194 135.502
6       20      134.58  1.17325 128.543 135.502
7       23      135.075 0.44733 134.549 136.018
8       19      130.718 24.4923 -1      136.018
9       19      135.712 0.488261        133.55  136.018
10      20      131.378 24.5831         -1      136.018
11      19      135.847 0.639702        133.377 136.018
12      21      135.947 1.0292          132.011 139.613
13      16      127.243 34.2913         -1      139.613
14      17      136.378 1.07856         136.018 139.613
15      21      136.857 1.5206          136.018 139.613
16      20      137.881 1.75342         136.018 139.613
17      25      125.311 42.1119         -1      139.613
18      20      139.266 1.22636         133.473 139.613
19      19      134.92  25.2398         -1      139.613
20      21      134.887 25.2345         -1      139.613
Mejor fitness: 139.613474
Mejor individuo: [98.0306448749573, -99.40782072027488]
```
Los datos correspoden a: 
- __gen__ : número de generación. 
- __neval__ : número de evaluaciones.
- __avg__ : desempeño medio de la población 
- __min__: valor mínimo de la población 
- __max__ : valor máximo dela población. 

Ademas del mejor fitness y el mejor individuo. 

Debido a que el objeto hof se almacena como lista, se puede obtener cada individuo que forma parte de él avanzando con indices. 
Para ello, usamos el atributo

```
fitness.values
```

Es importante remarcar que, en este caso, sabemos el máximo de la función y estamos optimizando en bae a eso. Conocer el resultado óptimo puede hacer que pensemos que la función no opero como debería, pero hay que establecerse en el marco de pensamiento dónde no conocemos el valor. 


Se extraen las siguientes conclusiones: 

- A lo largo de las generaciones, el valor medio (avg) se va incrementando debido a la mutación de los individuos. 
- La desviación típica (std) va disminuyendo. Esto indica que los individuos cada vez son más parecidos debido al elitismo por el proceso de selección. 
- El valor máaximo (max) va aumentando a lo largo de las generaciones. Muy positivo ys que es el indicar de buen funcionamiento. 

La columna __nevals__ no condice con el número de individuos de la población debido a las operaciones de cruce y mutación son probabiliísticas. Por ello, un individuo seleccionado no tiene por qué ser modificado. 

```
Todo individuo que, aun siendo seleccionado, no haya participado de ninguna operación genética, no será evaluado, ahorrando así operaciones redundantes.
```

Un aspecto que no está optimizado en deap es que si dos individuos son iguales no se evalúen dos veces. Esto es así porque el atributo que hace que un indivudo sea evaluado o no es la validez de su fitness, __atributo valid del fitness__. Al ser modificado, su valid pasa a false. 

Todos los individuos con un fitness invalido han de ser evaluados. Esto se realiza mediante el eaSimple. Por lo tanto, aunque a lo largo de la distintas generaciones se generen individuos iguales, estos serán evaluados de nuevo. 

