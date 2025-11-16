import numpy as np
import matplotlib.pyplot as plt

def monte_carlo_option_pricing(S0, K, r, sigma, T, N):
    
    Z = np.random.normal(0, 1, N)

    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)

    expected_price = np.mean(ST)
    sd_price = np.std(ST)

    call_price = np.exp(-r * T) * np.mean(np.maximum(ST - K, 0))
    put_price = np.exp(-r * T) * np.mean(np.maximum(K - ST, 0))

    plt.hist(ST, bins=50)
    plt.axvline(expected_price, color='green', linewidth=2)
    plt.title("Histogram of Simulated Stock Prices")
    plt.xlabel("Stock Price")
    plt.ylabel("Frequency")
    plt.show()

    return expected_price, sd_price, call_price, put_price


S0 = 100
K = 105
r = 0.05
sigma = 0.20
T = 1
N = 10000

result = monte_carlo_option_pricing(S0, K, r, sigma, T, N)
print("\nEXPECTED RESULTS:")
print("Expected stock price:", result[0])
print("Standard deviation:", result[1])
print("Call option price:", result[2])
print("Put option price:", result[3])



# ----- PART 3: Test different volatility levels -----
for sigma_test in [0.10, 0.20, 0.30]:
    ST = S0 * np.exp((r - 0.5 * sigma_test**2) * T + sigma_test * np.sqrt(T) * np.random.normal(0, 1, N))
    plt.hist(ST, bins=40, alpha=0.5, label=f"σ={sigma_test}")

plt.title("Comparison of Volatility Levels")
plt.xlabel("Stock Price")
plt.ylabel("Frequency")
plt.legend()
plt.show()





