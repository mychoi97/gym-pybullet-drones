import numpy as np
import matplotlib.pyplot as plt

# 업로드된 evaluations.npz 파일 로드
data = np.load("./results/save-01.12.2024_15.59.42/evaluations.npz")

print("data: ", data)

# 타임스탬프와 평균 보상 추출
timesteps = data['timesteps']
mean_rewards = np.mean(data['results'], axis=1)
# episode_length = np.mean(data['ep_lengths'], axis=1)

# 시각화
plt.figure(figsize=(10, 5))
plt.plot(timesteps, mean_rewards, marker='o', color='b', label='Mean Reward')
# plt.plot(timesteps, episode_length, marker='o', color='b', label='Episode Length')
plt.title("Learning Progress Over Time")
plt.xlabel("Timesteps")
plt.ylabel("Mean Reward")
# plt.ylabel("Episode Length")
plt.grid(True)
plt.legend()
plt.show()
