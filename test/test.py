import CustomDataset


a = CustomDataset.label_transform_tensor("ABC98")
print(a)
print(CustomDataset.tensor_transform_label(a))
