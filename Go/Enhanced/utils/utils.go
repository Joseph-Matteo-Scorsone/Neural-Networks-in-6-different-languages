package utils

func ReLU(x float64) float64 {
	if x > 0 {
		return x
	}
	return 0
}

func ReLUDeriv(x float64) float64 {
	if x > 0 {
		return 1
	}
	return 0
}

func MSE(y, yPred float64) float64 {
	return 0.5 * (y - yPred) * (y - yPred)
}

func MSEDeriv(y, yPred float64) float64 {
	return yPred - y
}
