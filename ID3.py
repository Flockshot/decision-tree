import numpy as np


# In the decision tree, non-leaf nodes are going to be represented via TreeNode
class TreeNode:
    def __init__(self, attribute):
        self.attribute = attribute
        # dictionary, k: subtree, key (k) an attribute value, value is either TreeNode or TreeLeafNode
        self.subtrees = {}


# In the decision tree, leaf nodes are going to be represented via TreeLeafNode
class TreeLeafNode:
    def __init__(self, data, label):
        self.data = data
        self.labels = label


class DecisionTree:
    def __init__(self, dataset: list, labels, features, criterion="information gain"):
        """
        :param dataset: array of data instances, each data instance is represented via an Python array
        :param labels: array of the labels of the data instances
        :param features: the array that stores the name of each feature dimension
        :param criterion: depending on which criterion ("information gain" or "gain ratio") the splits are to be performed
        """
        self.dataset = dataset
        self.labels = labels
        self.features = features
        self.criterion = criterion
        # it keeps the root node of the decision tree
        self.root = None

        # further variables and functions can be added...

    def calculate_entropy__(self, dataset, labels):
        """
        :param dataset: array of the data instances
        :param labels: array of the labels of the data instances
        :return: calculated entropy value for the given dataset
        """
        entropy_value = 0.0

        """
        Entropy calculations
        """
        label_count = {}

        for label in labels:
            label_count[label] = label_count.get(label, 0) + 1

        for label in label_count.keys():
            p = label_count[label] / len(labels)
            entropy_value -= p * np.log2(p)

        return entropy_value

    def calculate_average_entropy__(self, dataset, labels, attribute):
        """
        :param dataset: array of the data instances on which an average entropy value is calculated
        :param labels: array of the labels of those data instances
        :param attribute: for which attribute an average entropy value is going to be calculated...
        :return: the calculated average entropy value for the given attribute
        """
        average_entropy = 0.0
        """
            Average entropy calculations
        """

        attributes = np.array(dataset)[:, self.features.index(attribute)]
        unique_attr = set(attributes)

        for attr_val in unique_attr:

            attr_labels = []

            for i in range(len(labels)):
                if attributes[i] == attr_val:
                    attr_labels.append(labels[i])

            # formula for average entropy
            average_entropy += self.calculate_entropy__(None, attr_labels) * len(attr_labels) / len(labels)

        return average_entropy

    def calculate_information_gain__(self, dataset, labels, attribute):
        """
        :param dataset: array of the data instances on which an information gain score is going to be calculated
        :param labels: array of the labels of those data instances
        :param attribute: for which attribute the information gain score is going to be calculated...
        :return: the calculated information gain score
        """
        information_gain = 0.0
        """
            Information gain calculations
        """

        information_gain = self.calculate_entropy__(None, labels) - self.calculate_average_entropy__(dataset, labels,
                                                                                                     attribute)

        return information_gain

    def calculate_intrinsic_information__(self, dataset, labels, attribute):
        """
        :param dataset: array of data instances on which an intrinsic information score is going to be calculated
        :param labels: array of the labels of those data instances
        :param attribute: for which attribute the intrinsic information score is going to be calculated...
        :return: the calculated intrinsic information score
        """
        intrinsic_info = 0.0
        """
            Intrinsic information calculations for a given attribute
        """

        attributes = np.array(dataset)[:, self.features.index(attribute)]
        unique_attr = set(attributes)

        for attr_val in unique_attr:
            attr_count = 0
            for attr in attributes:
                if attr == attr_val:
                    attr_count += 1

            p = attr_count / len(labels)
            intrinsic_info -= p * np.log2(p)

        return intrinsic_info

    def calculate_gain_ratio__(self, dataset, labels, attribute):
        """
        :param dataset: array of data instances with which a gain ratio is going to be calculated
        :param labels: array of labels of those instances
        :param attribute: for which attribute the gain ratio score is going to be calculated...
        :return: the calculated gain ratio score
        """
        """
            Your implementation
        """

        gain_ratio = self.calculate_information_gain__(dataset, labels,
                                                       attribute) / self.calculate_intrinsic_information__(dataset,
                                                                                                           labels,
                                                                                                           attribute)

        return gain_ratio

    def ID3__(self, dataset, labels, used_attributes):
        """
        Recursive function for ID3 algorithm
        :param dataset: data instances falling under the current  tree node
        :param labels: labels of those instances
        :param used_attributes: while recursively constructing the tree, already used labels should be stored in used_attributes
        :return: it returns a created non-leaf node or a created leaf node
        """
        """
            Your implementation
        """

        if len(self.features) == len(used_attributes) or len(set(labels)) == 1:
            return TreeLeafNode(dataset, labels)

        # create a new list of unused features
        rest_attr = list(set(self.features).difference(used_attributes))

        # choosing the best attribute to split on based on the criterion
        if self.criterion == "information gain":
            attr = max(rest_attr, key=lambda attribute: self.calculate_information_gain__(dataset, labels, attribute))
        elif self.criterion == "gain ratio":
            attr = max(rest_attr, key=lambda attribute: self.calculate_gain_ratio__(dataset, labels, attribute))

        used_attributes.append(attr)

        # creating a tree on the best attribute
        tree = TreeNode(attr)

        # getting the index of the best attribute and the unique values of the best attribute
        attr_index = self.features.index(attr)
        unique_attr = set(np.array(dataset)[:, self.features.index(attr)])

        # creating a subtree for each value of the best attribute
        for attr_val in unique_attr:

            new_dataset = []
            new_labels = []
            for i in range(len(dataset)):
                if dataset[i][attr_index] == attr_val:
                    new_dataset.append(dataset[i])
                    new_labels.append(labels[i])

            # creating subtree and attach it to the tree
            subtree = self.ID3__(new_dataset, new_labels, used_attributes)
            tree.subtrees[attr_val] = subtree

        return tree

    def predict(self, x):
        """
        :param x: a data instance, 1 dimensional Python array 
        :return: predicted label of x
        
        If a leaf node contains multiple labels in it, the majority label should be returned as the predicted label
        """
        predicted_label = None
        """
            Your implementation
        """

        node = self.root

        # traversing the tree branches until we reach a leaf node
        while not isinstance(node, TreeLeafNode):
            # get the index of the attribute that the current node is split on
            attr_index = self.features.index(node.attribute)

            # get the value of the attribute from x to decide which branch to go to
            attr_val = x[attr_index]

            # follow the branch by updating the current node
            node = node.subtrees[attr_val]

        # after reaching a leaf node we will make the prediction based on the majority class in the leaf
        labels = node.labels
        label_count = {}

        for label in labels:
            label_count[label] = label_count.get(label, 0) + 1

        labels_sort = sorted(label_count, reverse=True)
        print(labels_sort)
        predicted_label = labels_sort[0]

        #classes = set(labels)
        #predicted_label = max(classes, key=labels.count)
        #print(predicted_label)

        return predicted_label

    def train(self):
        self.root = self.ID3__(self.dataset, self.labels, [])
        print("Training completed")
