import csv
import numpy as np
import matplotlib.pyplot as plt


anime_data = []
with open("Data/AnimeList.csv", "r", encoding="utf8") as csv_data:
	csv_reader = csv.reader(csv_data, delimiter=',')
	firstLine = True
	for row in csv_reader:
		if firstLine: firstLine = False
		else:
			anime_data.append(row)

anime_meta = [[int(a[0]), float(a[15]), int(a[16]), int(a[19])] for a in anime_data]

anime_members_ranked = sorted(anime_meta, key=lambda m: m[3], reverse=True)
anime_scoredby_ranked = sorted(anime_meta, key=lambda m: m[2], reverse=True)

scoredby = [m[2] for m in anime_meta]
print(max(scoredby), min(scoredby), np.mean(np.array(scoredby)), np.median(np.array(scoredby)))

# plt.subplot(1, 2, 1)
# plt.plot([m[3] for m in anime_members_ranked])
# plt.ylabel('people watching')
# plt.xlabel('ranked items')
# plt.title("Pareto's curve of items watchings")
# plt.grid(True)

# plt.subplot(1, 2, 2)
plt.plot([m[2] for m in anime_scoredby_ranked])
plt.ylabel('nratings')
plt.xlabel('ranked items')
plt.title("Pareto's curve of items ratings")
plt.grid(True)

plt.show()




# plt.scatter([m[3] for m in anime_meta], [m[2] for m in anime_meta])
# plt.ylabel('ratings')
# plt.xlabel('people watching')
# plt.title('Items ratings vs. items watching')
# plt.grid(True)
# plt.show()


