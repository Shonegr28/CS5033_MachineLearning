import matplotlib.pyplot as plt


"""
FILE: calculates and plots results from the neural network experiments
"""


def average_results(results):

    # stores averages
    average_training_accuracy = 0.0
    average_validation_accuracy = 0.0
    average_final_cost = 0.0

    # add metrics from every repetition
    for result in results:
        average_training_accuracy += result[0]
        average_validation_accuracy += result[1]
        average_final_cost += result[2]

    # divide by number of repetitions
    average_training_accuracy /= len(results)
    average_validation_accuracy /= len(results)
    average_final_cost /= len(results)


    # number of epochs stored in cost history
    num_epochs = len(results[0][3])

    # initialize average cost for each epoch
    average_costs = []

    for epoch in range(num_epochs):

        total_cost = 0.0

        # add cost at this epoch from every repetition
        for result in results:
            total_cost += result[3][epoch]

        # average cost at this epoch
        average_cost = total_cost / len(results)

        average_costs.append(average_cost)


    return (
        average_training_accuracy,
        average_validation_accuracy,
        average_final_cost,
        average_costs
    )


def print_results(dataset_names, all_results):

    print("\n\n========================================")
    print("AVERAGE RESULTS")
    print("========================================")

    for i in range(len(dataset_names)):

        averages = average_results(all_results[i])

        print("\nDataset:", dataset_names[i])
        print("Average Training Accuracy:", averages[0])
        print("Average Validation Accuracy:", averages[1])
        print("Average Final Cost:", averages[2])


def plot_validation_accuracy(dataset_names, all_results):
    validation_accuracies = []

    for results in all_results:
        averages = average_results(results)
        validation_accuracies.append(averages[1])

    plt.figure(figsize=(12, 6))
    plt.bar(dataset_names, validation_accuracies)

    plt.xlabel("Dataset")
    plt.ylabel("Average Validation Accuracy")
    plt.title("Average Validation Accuracy Across 10 Repetitions")
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig("validation_accuracy.png")
    plt.show()


def plot_average_cost(dataset_names, all_results):
    plt.figure(figsize=(12, 6))

    for i in range(len(dataset_names)):
        averages = average_results(all_results[i])
        average_costs = averages[3]

        epochs = []
        for epoch in range(len(average_costs)):
            epochs.append(epoch)

        plt.plot(epochs, average_costs, label=dataset_names[i])

    plt.xlabel("Epoch")
    plt.ylabel("Average Cost")
    plt.title("Average Cost Across 10 Repetitions")
    plt.legend()

    plt.tight_layout()
    plt.savefig("average_cost.png")
    plt.show()




import csv


def save_results_csv(dataset_names, all_results):

    with open("results_data.csv", "w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow([
            "Dataset",
            "Repetition",
            "Epoch",
            "Cost",
            "Training Accuracy",
            "Validation Accuracy"
        ])

        for i in range(len(dataset_names)):
            for repetition in range(len(all_results[i])):

                result = all_results[i][repetition]

                training_accuracy = result[0]
                validation_accuracy = result[1]
                costs = result[3]

                for epoch in range(len(costs)):
                    writer.writerow([
                        dataset_names[i],
                        repetition + 1,
                        epoch,
                        costs[epoch],
                        training_accuracy,
                        validation_accuracy
                    ])