'''
Viva Questions:-



'''



import seaborn as sns
import matplotlib.pyplot as plt
from seaborn import load_dataset

# Load the Titanic dataset
titanic = sns.load_dataset('titanic')

# Display the first few rows of the dataset
print(titanic.head())

# Plot histogram of ticket fares
plt.figure(figsize=(10, 6))

# kde - kernel density estimate
sns.histplot(titanic['age'], bins=30, kde=True, color='red')

plt.title('Distribution of Ticket Fares')
plt.xlabel('Fare')
plt.ylabel('Frequency')
plt.show()

