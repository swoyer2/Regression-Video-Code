from manim import *
import pandas as pd
import random

data = pd.read_csv('avocadoNew.csv', sep= ";", header='infer',)
X = data['AveragePrice']
y = data['Total Volume']

class RegressionExample(Scene):
    def construct(self):
        self.intro()
        self.clear()
        self.whatIsRegression()
        self.clear()
        self.squareError()
        self.clear()
        self.showNormalEquation()
        self.clear()
        self.comparingFunctions()
        self.clear()
        self.square8020()
        self.clear()
        self.comparingFunctions2()
        self.clear()
        self.showAverages()
        self.clear()
        self.finale()
    
    def clear(self):
        self.play(
            *[FadeOut(mob)for mob in self.mobjects]
            # All mobjects in the screen are saved in self.mobjects
        )

    def intro(self):
        title = Text("Avocado Prices vs. Total Volume Sold in Los Angeles").scale(0.7).to_edge(UP)
        self.play(Write(title))
        
        # Create the axes
        axes = Axes(
            x_range=[0.4, 2.2, 0.2],  # Range for Average Price
            y_range=[.8, 5.6, 0.5],  # Range for Total Volume (in millions)
            axis_config={"include_numbers": True}
        )
        
        labels = axes.get_axis_labels(
            Text("Price per Avocado ($)").scale(0.4),
            Text("Volume Sold (in million lbs)").scale(0.4)
        )
        
        # Add the axes and labels to the scene
        self.play(FadeOut(title), Create(axes), Write(labels))
        data_points = []

        for i, x in enumerate(X):
            data_points.append((x, y[i]/1000000))
        points = [axes.coords_to_point(x, y) for x, y in data_points]
        dots = VGroup(*[Dot(point, color=BLUE) for point in points])
        self.play(LaggedStart(*[FadeIn(dot) for dot in dots], lag_ratio=0.05, run_time=4))

        # Create the $2 point
        twoDollarDot = Dot(axes.coords_to_point(2,5), color=YELLOW)
        twoDollarDot.z_index = 10
        self.play(FadeIn(twoDollarDot))

        # Define the quadratic function (as an example)
        quadraticWeights = [6069187.59438185, -4446266.79066955,  1183399.69655964]
        scaledWeights = []
        for weight in quadraticWeights:
            scaledWeights.append(weight/1000000)

        tempfunctions = [lambda x: 5 - 2*x, 
                         lambda x: 4.623786 - 1.738635*x, 
                         lambda x: 1+1.8*x,
                         lambda x: 3+2.1*x-x**2,
                         lambda x: 2+2.1*x-0.5*x**2,
                         lambda x: 2+1.8*x-x**2]
        firstFunction = axes.plot(tempfunctions[0], x_range=[0.4, 2], color=RED)
        self.play(Create(firstFunction), Transform(twoDollarDot, Dot(axes.coords_to_point(2,tempfunctions[0](2)), color=YELLOW)))
        self.play(Transform(firstFunction, axes.plot(tempfunctions[1], x_range=[0.4, 2], color=RED)), Transform(twoDollarDot, Dot(axes.coords_to_point(2,tempfunctions[1](2)), color=YELLOW)))
        self.play(Transform(firstFunction, axes.plot(tempfunctions[2], x_range=[0.4, 2], color=RED)), Transform(twoDollarDot, Dot(axes.coords_to_point(2,tempfunctions[2](2)), color=YELLOW)))
        self.play(Transform(firstFunction, axes.plot(tempfunctions[3], x_range=[0.4, 2], color=RED)), Transform(twoDollarDot, Dot(axes.coords_to_point(2,tempfunctions[3](2)), color=YELLOW)))
        self.play(Transform(firstFunction, axes.plot(tempfunctions[4], x_range=[0.4, 2], color=RED)), Transform(twoDollarDot, Dot(axes.coords_to_point(2,tempfunctions[4](2)), color=YELLOW)))
        self.play(Transform(firstFunction, axes.plot(tempfunctions[5], x_range=[0.4, 2], color=RED)), Transform(twoDollarDot, Dot(axes.coords_to_point(2,tempfunctions[5](2)), color=YELLOW)))

        # Keep the scene displayed for a moment
        self.clear()

        code = '''import pandas as pd

data = pd.read_csv('avocadoNew.csv', sep= ";", header='infer',)
data['Total Volume'] /= 1000000
'''
        rendered_code = Code(code=code, tab_width=4, background="window",
                            language="Python", font="Monospace", style=Code.styles_list[15])
        self.play(Write(rendered_code))
        self.wait(2)
    
    def whatIsRegression(self):
        # Define the linear function
        linear_func = MathTex("f(x) = w_0 + w_1 x").scale(1.5)

        # Display the linear function
        self.play(Write(linear_func))
        self.wait(1)

        # Emphasize x (index 5 for 'x')
        x = linear_func[0][10]
        self.play(Indicate(x))
        self.wait(1)

        # Emphasize f(x) (index 0 for 'f(x)')
        f_x = linear_func[0][0:4]
        self.play(Indicate(f_x))
        self.wait(1)

        # Emphasize w_0 and w_1 at the same time (index 2 for 'w_0' and index 4 for 'w_1')
        w_0 = linear_func[0][5:7]
        w_1 = linear_func[0][8:10]
        self.play(Indicate(w_0), Indicate(w_1))
        self.wait(1)

        randomWeights = [random.randrange(-30, 30)/10 for i in range(10)]
        randomWeights2 = [random.randrange(0, 30)/10 for i in range(10)]
        for i, randomWeight in enumerate(randomWeights):
            self.play(Transform(linear_func, MathTex(f"f(x) = {randomWeight} + {randomWeights2[i]} x").scale(1.5), run_time=0.25))


        # Show a quadratic function
        quadratic_func = MathTex("f(x) = w_0 + w_1 x + w_2 x^2").scale(1.5)
        self.play(Transform(linear_func, quadratic_func))
        self.wait(1)

        # Show a cubic function
        cubic_func = MathTex("f(x) = w_0 + w_1 x + w_2 x^2 + w_3 x^3").scale(1.5)
        self.play(Transform(linear_func, cubic_func))
        self.wait(1)

        # Show a higher-order function
        high_order_func = MathTex("f(x) = w_0 + w_1 x + w_2 x^2 + ... + w_n x^n").scale(1.5)
        self.play(Transform(linear_func, high_order_func))

        self.clear()

        code = '''def linear(w0, w1):
    return lambda x : w0 + w1*x

def quadratic(w0, w1, w2):
    return lambda x : w0 + w1*x + w2*(x**2)

def cubic(w0, w1, w2, w3):
    return lambda x : w0 + w1*x + w2*(x**2) + w3*(x**3)

def quartic(w0, w1, w2, w3, w4):
    return lambda x : w0 + w1*x + w2*(x**2) + w3*(x**3) + w4*(x**4)
'''
        rendered_code = Code(code=code, tab_width=4, background="window",
                            language="Python", font="Monospace", style=Code.styles_list[15]).scale(0.8)
        self.play(Write(rendered_code))
        self.wait(2)
        self.wait(2)
    
    def squareError(self):
        # Create the axes
        axes = Axes(
            x_range=[0.4, 2.2, 0.2],  # Range for Average Price
            y_range=[.8, 5.6, 0.5],  # Range for Total Volume (in millions)
            axis_config={"include_numbers": True}
        )
        
        labels = axes.get_axis_labels(
            Text("Price per Avocado ($)").scale(0.4),
            Text("Volume Sold (in million lbs)").scale(0.4)
        )
        self.play(Create(axes), Write(labels))
        data_points = []

        for i, x in enumerate(X):
            data_points.append((x, y[i]/1000000))
        points = [axes.coords_to_point(x, y) for x, y in data_points]
        dots = VGroup(*[Dot(point, color=BLUE) for point in points])
        self.play(FadeIn(dots))

        def getDifferenceLine(x, y, function):
            l = Line(Dot(axes.coords_to_point(x, y), radius=0), Dot(axes.coords_to_point(x, function(x)), radius=0), color=YELLOW)
            l.set_z_index = -1
            return l, (function(x) - y)**2
        
        def countAllLines(function, color, position):
            lines = VGroup()
            firstLine, lineLength = getDifferenceLine(data_points[0][0], data_points[0][1], function)
            count = DecimalNumber(lineLength, color=color).shift(UP * 3 + position)

            if color == RED:
                lineFunction = axes.plot(function, x_range=[0.4, 2], color=color)
            else:
                lineFunction = axes.plot(function, x_range=[0.4, 1.68], color=color)
            self.play(Create(lineFunction))

            self.play(Create(firstLine))
            self.play(FadeIn(count))
            totalError = lineLength
            for dataPoint in data_points[1:]:
                line, lineLength = getDifferenceLine(dataPoint[0], dataPoint[1], function)
                lineLength = lineLength
                lines.add(line)
                totalError += lineLength
            
            self.play(Create(lines), FadeOut(firstLine))
            totalCount = DecimalNumber(totalError, color=color).shift(UP * 3 + position)
            self.play(Transform(lines, totalCount), FadeOut(count))
        
        countAllLines(lambda x: 5 - 2*x, RED, np.array((5.0, 0.0, 0.0)))
        countAllLines(lambda x: 5 - 2.5*x, GREEN, np.array((3.0, 0.0, 0.0)))

        equation = MathTex(
            "SE", "=", "\\sum_{i=1}^{n}", "(y_i", "-", "\\hat{y}_i)^2"
        ).shift(np.array((3.0, 2.0, 0.0)))
        self.play(Write(equation))

        self.clear()

        code = '''import numpy as np

def costFunction(predictedValues, actualValues):
    predictedValues = np.array(predictedValues)
    actualValues = np.array(actualValues)
    differencesSquared = (predictedValues - actualValues) ** 2
    return sum(differencesSquared)
'''
        rendered_code = Code(code=code, tab_width=4, background="window",
                            language="Python", font="Monospace", style=Code.styles_list[15])
        self.play(Write(rendered_code))

        self.wait(2)

    def showNormalEquation(self):
        theta = MathTex(r"\theta = (X^T X)^{-1} X^T y").scale(1.5)

        self.play(Write(theta))
        self.wait(2)
        W_0 = f"w_0"
        W_1 = f"w_1"
        W_2 = f"w_2"
        weights = Matrix([[W_0, W_1, W_2]]).shift(DOWN)
        X = Matrix([["x_1^0","x_1^1" ,"x_1^2","...", "x_1^n"], 
                    ["x_2^0","x_2^1" ,"x_2^2","...", "x_2^n"],
                    ["x_3^0","x_3^1" ,"x_3^2","...", "x_3^n"],
                    [r"&\vdots\\", r"&\vdots\\", r"&\vdots\\", r"&\ddots\\", r"&\vdots\\"],
                    ["x_m^0","x_m^1" ,"x_m^2","...", "x_m^n"]]).shift(DOWN)
        y = Matrix([["y_0", "y_1", "y_2", "...", "y_m"]]).shift(DOWN)
        XTransposed = Matrix([["x_1^0","x_2^0" ,"x_3^0","...", "x_m^0"], 
                    ["x_1^1","x_2^1" ,"x_3^1","...", "x_m^1"],
                    ["x_1^2","x_2^2" ,"x_3^2","...", "x_m^2"],
                    [r"&\vdots\\", r"&\vdots\\", r"&\vdots\\", r"&\ddots\\", r"&\vdots\\"],
                    ["x_1^n","x_2^n" ,"x_3^n","...", "x_m^n"]]).shift(DOWN)

        self.play(Indicate(theta[0][0]), FadeIn(weights))
        self.wait(1)
        self.play(FadeOut(weights))
        self.play(Indicate(theta[0][5]), Indicate(theta[0][3]), Indicate(theta[0][-3]))
        self.play(FadeIn(X), theta.animate.shift(2*UP))
        self.wait(1)
        self.play(FadeOut(X), theta.animate.shift(2*DOWN))
        self.play(Indicate(theta[0][-1]), FadeIn(y))
        self.wait(1)
        self.play(FadeOut(y))
        self.play(Indicate(theta[0][4]), Indicate(theta[0][-2]))
        self.play(FadeIn(X), theta.animate.shift(2*UP))
        self.wait(1)
        self.play(Transform(X, XTransposed))
        self.wait(1)
        self.play(FadeOut(X), theta.animate.shift(2*DOWN))
        self.wait(1)

        self.clear()

        code = '''
def normalEquation(X, y):
    XTranspose = np.transpose(X)
    XTransposeX = np.dot(XTranspose, X)
    XTransposey = np.dot(XTranspose, y) 
    return np.linalg.solve(XTransposeX, XTransposey)
'''
        rendered_code = Code(code=code, tab_width=4, background="window",
                            language="Python", font="Monospace", style=Code.styles_list[15])
        self.play(Write(rendered_code))

        self.wait(2)

    def comparingFunctions(self):
        # Create the axes
        axes = Axes(
            x_range=[0.4, 2.2, 0.2],  # Range for Average Price
            y_range=[.8, 5.6, 0.5],  # Range for Total Volume (in millions)
            axis_config={"include_numbers": True}
        )
        
        labels = axes.get_axis_labels(
            Text("Price per Avocado ($)").scale(0.4),
            Text("Volume Sold (in million lbs)").scale(0.4)
        )
        
        # Add the axes and labels to the scene
        self.play(Create(axes), Write(labels))
        data_points = []

        for i, x in enumerate(X):
            data_points.append((x, y[i]/1000000))
        points = [axes.coords_to_point(x, y) for x, y in data_points]
        dots = VGroup(*[Dot(point, color=BLUE) for point in points])
        self.play(LaggedStart(*[FadeIn(dot) for dot in dots], lag_ratio=0.05, run_time=4))

        # # Create the $2 point
        # twoDollarDot = Dot(axes.coords_to_point(2,5), color=YELLOW)
        # twoDollarDot.z_index = 10
        # self.play(FadeIn(twoDollarDot))

        # Define the quadratic function (as an example)
        quadraticWeights = [6069187.59438185, -4446266.79066955,  1183399.69655964]
        scaledWeights = []
        for weight in quadraticWeights:
            scaledWeights.append(weight/1000000)

        tempfunctions = [lambda x: 4.62378625348872 - 1.73863534334986*x, 
                         lambda x: 6.06918759438185 - 4.44626679066955*x + 1.18339969655964*x**2, 
                         lambda x: 11.2799438521901 - 19.65435312493806*x + 15.22025341756894*x**2 - 4.06967685753312*x**3,
                         lambda x: 29.1899724 - 90.4642759*x + 115.902323*x**2 - 65.1072266*x**3 + 13.3196133*x**4]
        linearFunction = axes.plot(tempfunctions[0], x_range=[0.4, 2], color=YELLOW)
        quadraticFunction = axes.plot(tempfunctions[1], x_range=[0.4, 2], color=RED)
        cubicFunction = axes.plot(tempfunctions[2], x_range=[0.4, 2], color=GREEN)
        fourthFunction = axes.plot(tempfunctions[3], x_range=[0.5, 2], color=PURPLE)
        error = Text(f"Error: 31.71").shift(3*UP + 3*RIGHT).scale(0.5)
        self.play(Create(linearFunction), Create(error))
        self.play(Uncreate(linearFunction), Create(quadraticFunction), Transform(error, Text(f"Error: 29.73").shift(3*UP + 3*RIGHT).scale(0.5)))
        self.play(Uncreate(quadraticFunction), Create(cubicFunction), Transform(error, Text(f"Error: 27.69").shift(3*UP + 3*RIGHT).scale(0.5)))
        self.play(Uncreate(cubicFunction), Create(fourthFunction), Transform(error, Text(f"Error: 25.18").shift(3*UP + 3*RIGHT).scale(0.5)))
        # Keep the scene displayed for a moment
        self.wait(2)
    
    def square8020(self):
        def create_squares():
            squares = VGroup()
            for i in range(5):
                square = Square(side_length=1, fill_color=WHITE, fill_opacity=1)
                square.set_stroke(BLACK)
                square.move_to(RIGHT * i)
                squares.add(square)
            return squares

        def shade_square(square):
            return square.animate.set_fill(ORANGE, opacity=1)

        squares = create_squares().center()
        # Create and position the "Training data" text
        training_text = Text("Training data").next_to(squares, UP).scale(0.5)
        self.play(Create(squares), Write(training_text))
        
        square_to_shade = squares[0]
        self.play(shade_square(square_to_shade))

        # Move "Training data" text and replace it with "Testing data"
        testing_text = Text("Testing data", color=ORANGE).next_to(square_to_shade, UP).scale(0.5)
        self.play(
            training_text.animate.shift(RIGHT),
            Write(testing_text)
        )
        self.play(FadeOut(testing_text), FadeOut(training_text))
        self.play(
            square_to_shade.animate.move_to(squares[1].get_center()),
            squares[1].animate.move_to(square_to_shade.get_center())
        )
        self.play(
            square_to_shade.animate.move_to(squares[2].get_center()),
            squares[2].animate.move_to(square_to_shade.get_center())
        )
        self.play(
            square_to_shade.animate.move_to(squares[3].get_center()),
            squares[3].animate.move_to(square_to_shade.get_center())
        )
        self.play(
            square_to_shade.animate.move_to(squares[4].get_center()),
            squares[4].animate.move_to(square_to_shade.get_center())
        )
        self.wait(2)
    
    def comparingFunctions2(self):
        # Create the axes
        axes = Axes(
            x_range=[0.4, 2.2, 0.2],  # Range for Average Price
            y_range=[.8, 5.6, 0.5],  # Range for Total Volume (in millions)
            axis_config={"include_numbers": True}
        )
        
        labels = axes.get_axis_labels(
            Text("Price per Avocado ($)").scale(0.4),
            Text("Volume Sold (in million lbs)").scale(0.4)
        )
        labels.z_index = 20
        
        # Add the axes and labels to the scene
        self.play(Create(axes), Write(labels))

        fifth = len(X) // 5

        # Create training data that excludes one segment for testing
        trainingData = []
        testingData = []

        for i in range(5):
            # Define the testing segment
            test_start = i * fifth
            test_end = test_start + fifth
            
            # Create the testing data segment
            testingData.append((X[test_start:test_end], y[test_start:test_end]))
            
            # Create the training data segment by excluding the current test segment
            train_X = X[:test_start].tolist() + X[test_end:].tolist()
            train_y = y[:test_start].tolist() + y[test_end:].tolist()
            
            trainingData.append((train_X, train_y))

        def displayPoints(data, color, previousPoints=None):
            data_points = [(x, y / 1_000_000) for x, y in zip(data[0], data[1])]
            points = [axes.coords_to_point(x, y) for x, y in data_points]
            dots = VGroup(*[Dot(point, color=color) for point in points])
            
            if previousPoints is None:
                self.play(LaggedStart(*[FadeIn(dot) for dot in dots], lag_ratio=0.05, run_time=1))
            else:
                self.play(Transform(previousPoints, dots))
            
            return dots

        linearfunctions = [lambda x: 4.82678672 - 1.86943009*x, 
                           lambda x: 4.63901374 - 1.69091732*x,
                           lambda x: 4.48873795 - 1.63003694*x,
                           lambda x: 5.13607215 - 2.327164*x,
                           lambda x: 4.34224882 - 1.53662989*x
                           ]
        
        quadraticfunctions = [lambda x: 6.32493397 - 4.67833289*x + 1.22317661*x**2, 
                           lambda x: 5.90945864 - 4.06781771*x + 1.03218484*x**2,
                           lambda x: 5.85379784 - 4.12129389*x + 1.06294675*x**2,
                           lambda x: 6.62156469 - 5.51141143*x + 1.6545134*x**2,
                           lambda x: 5.54685198 - 3.75628113*x + 0.94794745*x**2
                           ]
        
        cubicfunctions = [lambda x: 10.62239213 - 17.2149043*x + 12.75752348*x**2 - 3.33221476*x**3, 
                          lambda x: 12.08339412 - 22.2543087*x + 17.91211274*x**2 - 4.90083611*x**3,
                          lambda x: 11.08577656 - 19.15972982*x + 14.75943637*x**2 - 3.92938577*x**3,
                          lambda x: 18.87577388 - 44.64967134*x + 41.76516759*x**2 - 13.17788213*x**3,
                          lambda x: 9.74544797 - 15.70840827*x + 11.75136562*x**2 - 3.08111164*x**3]

        fourthfunctions = [lambda x: 26.53832558 - 80.2103472*x + 102.31626277*x**2 - 57.57626612*x**3 + 11.82213702*x**4, 
                           lambda x: 27.9533615 - 85.48214029*x + 108.37825742*x**2 - 60.02533279*x**3 + 12.07521788*x**4,
                           lambda x: 29.24075558 - 90.41195646*x + 115.26927243*x**2 - 64.38496264*x**3 + 13.09786506*x**4,
                           lambda x: 42.87752115 - 149.75632639*x + 208.74260059*x**2 - 127.27179666*x**3 + 28.28133901*x**4,
                           lambda x: 27.07938694 - 82.02687272*x + 103.34401858*x**2 - 57.18406284*x**3 + 11.54273138*x**4]
        
        linearLines = [axes.plot(linearfunctions[0], x_range=[0.4, 2], color=YELLOW),
                       axes.plot(linearfunctions[1], x_range=[0.4, 2], color=YELLOW),
                       axes.plot(linearfunctions[2], x_range=[0.4, 2], color=YELLOW),
                       axes.plot(linearfunctions[3], x_range=[0.4, 2], color=YELLOW),
                       axes.plot(linearfunctions[4], x_range=[0.4, 2], color=YELLOW)]
        
        quadraticLines = [axes.plot(quadraticfunctions[0], x_range=[0.4, 2], color=RED),
                          axes.plot(quadraticfunctions[1], x_range=[0.4, 2], color=RED),
                          axes.plot(quadraticfunctions[2], x_range=[0.4, 2], color=RED),
                          axes.plot(quadraticfunctions[3], x_range=[0.4, 2], color=RED),
                          axes.plot(quadraticfunctions[4], x_range=[0.4, 2], color=RED),]
        
        cubicLines = [axes.plot(cubicfunctions[0], x_range=[0.4, 2], color=GREEN),
                    axes.plot(cubicfunctions[1], x_range=[0.4, 2], color=GREEN),
                    axes.plot(cubicfunctions[2], x_range=[0.4, 2], color=GREEN),
                    axes.plot(cubicfunctions[3], x_range=[0.4, 2], color=GREEN),
                    axes.plot(cubicfunctions[4], x_range=[0.4, 2], color=GREEN),]

        fourthLines = [axes.plot(fourthfunctions[0], x_range=[0.4, 2], color=PURPLE),
                    axes.plot(fourthfunctions[1], x_range=[0.4, 2], color=PURPLE),
                    axes.plot(fourthfunctions[2], x_range=[0.4, 2], color=PURPLE),
                    axes.plot(fourthfunctions[3], x_range=[0.4, 2], color=PURPLE),
                    axes.plot(fourthfunctions[4], x_range=[0.4, 2], color=PURPLE),]
        
        linearErrors = [9.76825, 7.1856, 4.0071, 8.8463, 13.8903]
        quadraticErrors = [9.64191,  6.5136,   3.17978,   3.86571,  12.8524]
        cubicErrors = [8.70871,  7.14882,  2.74073,  98.4833,   11.5035]
        fourthErrors = [8.00551,  6.25673,  2.39448,  19.3876,    9.74568]

        errorTesting = Text(f"Error Testing: {linearErrors[0]}").shift(3*UP + 4*RIGHT).scale(0.5)
        a = displayPoints(trainingData[0], WHITE)
        b = displayPoints(testingData[0], ORANGE)
        self.play(Create(linearLines[0]), Write(errorTesting))
        self.play(Transform(linearLines[0], quadraticLines[0]), Transform(errorTesting, Text(f"Error Testing: {quadraticErrors[0]}").shift(3*UP + 4*RIGHT).scale(0.5)))
        self.play(Transform(linearLines[0], cubicLines[0]), Transform(errorTesting, Text(f"Error Testing: {cubicErrors[0]}").shift(3*UP + 4*RIGHT).scale(0.5)))
        self.play(Transform(linearLines[0], fourthLines[0]), Transform(errorTesting, Text(f"Error Testing: {fourthErrors[0]}").shift(3*UP + 4*RIGHT).scale(0.5)))
        
        displayPoints(trainingData[1], WHITE, a)
        displayPoints(testingData[1], ORANGE, b)
        self.play(Transform(linearLines[0], linearLines[1]), Transform(errorTesting, Text(f"Error Testing: {linearErrors[1]}").shift(3*UP + 4*RIGHT).scale(0.5)))
        self.play(Transform(linearLines[0], quadraticLines[1]), Transform(errorTesting, Text(f"Error Testing: {quadraticErrors[1]}").shift(3*UP + 4*RIGHT).scale(0.5)))
        self.play(Transform(linearLines[0], cubicLines[1]), Transform(errorTesting, Text(f"Error Testing: {cubicErrors[1]}").shift(3*UP + 4*RIGHT).scale(0.5)))
        self.play(Transform(linearLines[0], fourthLines[1]), Transform(errorTesting, Text(f"Error Testing: {fourthErrors[1]}").shift(3*UP + 4*RIGHT).scale(0.5)))

        displayPoints(trainingData[2], WHITE, a)
        displayPoints(testingData[2], ORANGE, b)
        self.play(Transform(linearLines[0], linearLines[2]), Transform(errorTesting, Text(f"Error Testing: {linearErrors[2]}").shift(3*UP + 4*RIGHT).scale(0.5)))
        self.play(Transform(linearLines[0], quadraticLines[2]), Transform(errorTesting, Text(f"Error Testing: {quadraticErrors[2]}").shift(3*UP + 4*RIGHT).scale(0.5)))
        self.play(Transform(linearLines[0], cubicLines[2]), Transform(errorTesting, Text(f"Error Testing: {cubicErrors[2]}").shift(3*UP + 4*RIGHT).scale(0.5)))
        self.play(Transform(linearLines[0], fourthLines[2]), Transform(errorTesting, Text(f"Error Testing: {fourthErrors[2]}").shift(3*UP + 4*RIGHT).scale(0.5)))

        displayPoints(trainingData[3], WHITE, a)
        displayPoints(testingData[3], ORANGE, b)
        self.play(Transform(linearLines[0], linearLines[3]), Transform(errorTesting, Text(f"Error Testing: {linearErrors[3]}").shift(3*UP + 4*RIGHT).scale(0.5)))
        self.play(Transform(linearLines[0], quadraticLines[3]), Transform(errorTesting, Text(f"Error Testing: {quadraticErrors[3]}").shift(3*UP + 4*RIGHT).scale(0.5)))
        self.play(Transform(linearLines[0], cubicLines[0]), Transform(errorTesting, Text(f"Error Testing: {cubicErrors[3]}").shift(3*UP + 4*RIGHT).scale(0.5)))
        self.play(Transform(linearLines[0], fourthLines[0]), Transform(errorTesting, Text(f"Error Testing: {fourthErrors[3]}").shift(3*UP + 4*RIGHT).scale(0.5)))

        displayPoints(trainingData[4], WHITE, a)
        displayPoints(testingData[4], ORANGE, b)
        self.play(Transform(linearLines[0], linearLines[4]), Transform(errorTesting, Text(f"Error Testing: {linearErrors[4]}").shift(3*UP + 4*RIGHT).scale(0.5)))
        self.play(Transform(linearLines[0], quadraticLines[4]), Transform(errorTesting, Text(f"Error Testing: {quadraticErrors[4]}").shift(3*UP + 4*RIGHT).scale(0.5)))
        self.play(Transform(linearLines[0], cubicLines[0]), Transform(errorTesting, Text(f"Error Testing: {cubicErrors[4]}").shift(3*UP + 4*RIGHT).scale(0.5)))
        self.play(Transform(linearLines[0], fourthLines[0]), Transform(errorTesting, Text(f"Error Testing: {fourthErrors[4]}").shift(3*UP + 4*RIGHT).scale(0.5)))

        self.clear()

        code = '''
def fiveFoldCrossValidation(X, y):
    costsTraining = []
    costsTest = []
    foldSize = len(y) // 5

    for i in range(5):
        # Split the data into training and testing sets
        testData = data['AveragePrice'].values[foldSize*i:foldSize*(i+1)]
        actualValues = y[foldSize*i:foldSize*(i+1)]
        
        trainingData = np.concatenate((X[:foldSize*i], X[foldSize*(i+1):]))
        trainingValues = np.concatenate((y[:foldSize*i], y[foldSize*(i+1):]))

        # Calculate weights using normal equation
        weights = normalEquation(trainingData, trainingValues)

        # Generate predictions for both training and testing data
        predictions = []
        for dataPoint in np.concatenate((data['AveragePrice'].values[:foldSize*i], data['AveragePrice'].values[foldSize*(i+1):], testData)):
            if len(weights) == 2:
                predictions.append(linear(weights[0], weights[1])(dataPoint))
            elif len(weights) == 3:
                predictions.append(quadratic(weights[0], weights[1], weights[2])(dataPoint))
            elif len(weights) == 4:
                predictions.append(cubic(weights[0], weights[1], weights[2], weights[3])(dataPoint))
            elif len(weights) == 5:
                predictions.append(quartic(weights[0], weights[1], weights[2], weights[3], weights[4])(dataPoint))

        # Calculate costs for training and testing data
        costsTraining.append(costFunction(predictions[:-foldSize], trainingValues))
        costsTest.append(costFunction(predictions[-foldSize:], actualValues))

    return costsTraining, costsTest

X = data['AveragePrice']
y = data['Total Volume']
linearData = fiveFoldCrossValidation(np.column_stack((np.ones(X.shape[0]), X)), y)
quadraticData = fiveFoldCrossValidation(np.column_stack((np.ones(X.shape[0]), X, X**2)), y)
cubicData = fiveFoldCrossValidation(np.column_stack((np.ones(X.shape[0]), X, X**2, X**3)), y)
quarticData = fiveFoldCrossValidation(np.column_stack((np.ones(X.shape[0]), X, X**2, X**3, X**4)), y)
'''
        rendered_code = Code(code=code, tab_width=4, background="window",
                            language="Python", font="Monospace", style=Code.styles_list[15]).scale(0.45)
        self.play(Write(rendered_code))
        self.wait(2)

    def showAverages(self):
        # Create the table
        table = Table(
            [['9.76825', '7.1856', '4.0071', '8.8463', '13.8903', '8.73951'],
             ['9.64191', '6.5136', '3.17978', '3.86571', '12.8524', '7.21067'],
             ['8.70871', '7.14882', '2.74073', '98.4833', '11.5035', '25.717'],
             ['8.00551', '6.25673', '2.39448', '19.3876', '9.74568', '9.15799']],
            [Text("Linear", color=YELLOW), Text("Quadratic", color=RED), Text("Cubic", color=GREEN), Text("Quartic", color=PURPLE)],
            [Text("Test 1"), Text("Test 2"), Text("Test 3"), Text("Test 4"), Text("Test 5"), Text("Average")],
            top_left_entry=SVGMobject('avocado.svg')
        ).scale(0.5)
        table.add(table.get_cell((3,7), color=RED))

        self.play(FadeIn(table))
        self.clear()

        code = '''
from tabulate import tabulate
table_data = [
    ['Linear'] + linearData[0] + linearData[1] + [np.mean(linearData[0])] + [np.mean(linearData[1])],
    ['Quadratic'] + quadraticData[0] + quadraticData[1] + [np.mean(quadraticData[0])] + [np.mean(quadraticData[1])],
    ['Cubic'] + cubicData[0] + cubicData[1] + [np.mean(cubicData[0])] + [np.mean(cubicData[1])],
    ['Quartic'] + quarticData[0] + quarticData[1] + [np.mean(quarticData[0])] + [np.mean(quarticData[1])],
]
headers = ['', '2345', '1345', '1245', '1235', '1234', '1', '2', '3', '4', '5', 'Mean for Training', 'Mean for Testing']

print(tabulate(table_data, headers=headers))
'''
        rendered_code = Code(code=code, tab_width=4, background="window",
                            language="Python", font="Monospace", style=Code.styles_list[15]).scale(0.45)
        self.play(Write(rendered_code))
        self.wait(2)
    
    def finale(self):
        # Create the axes
        axes = Axes(
            x_range=[0.4, 2.2, 0.2],  # Range for Average Price
            y_range=[.8, 5.6, 0.5],  # Range for Total Volume (in millions)
            axis_config={"include_numbers": True}
        )
        
        labels = axes.get_axis_labels(
            Text("Price per Avocado ($)").scale(0.4),
            Text("Volume Sold (in million lbs)").scale(0.4)
        )
        
        # Add the axes and labels to the scene
        self.play(Create(axes), Write(labels))
        data_points = []

        for i, x in enumerate(X):
            data_points.append((x, y[i]/1000000))
        points = [axes.coords_to_point(x, y) for x, y in data_points]
        dots = VGroup(*[Dot(point, color=WHITE) for point in points])
        self.play(LaggedStart(*[FadeIn(dot) for dot in dots], lag_ratio=0.05, run_time=1))

        finalFunction = lambda x: 6.06918759438185 - 4.44626679066955*x + 1.18339969655964*x**2
        # Create the $2 point
        twoDollarDot = Dot(axes.coords_to_point(2,finalFunction(2)), color=YELLOW)
        twoDollarDot.z_index = 10

        linearFunction = axes.plot(finalFunction, x_range=[0.4, 2], color=YELLOW)
        self.play(Create(linearFunction))
        self.play(FadeIn(twoDollarDot))
        self.play(Write(Text(str(round(finalFunction(2),4)) + ' Million Avocados Sold').shift(2*RIGHT + 3*UP).scale(0.5)))
        # Keep the scene displayed for a moment
        self.clear()
        code = '''
quadraticWeights = normalEquation(np.column_stack((np.ones(X.shape[0]), X, X**2)), y)
quadraticModel = quadratic(quadraticWeights[0], quadraticWeights[1], quadraticWeights[2])
def predict(price):
    return round(quadraticModel(price), 2)

print(f'Avocados sold at $2.00 in mil/lbs: {predict(2)}')
'''
        rendered_code = Code(code=code, tab_width=4, background="window",
                            language="Python", font="Monospace", style=Code.styles_list[15]).scale(0.6)
        self.play(Write(rendered_code))
        self.wait(2)
        self.wait(2)

