from .model_collections import ModelCollectionBuilder


def build_dataset(
    config,
    model_tags,
    category_tags,
    random_subset=1.0,
    cache_size=0
):
    # Create a dataset instance to generate the samples for training
    dataset_directory = config["data"]["dataset_directory"]
    dataset_type = config["data"]["dataset_type"]
    train_test_splits_file = config["data"]["splits_file"]
    dataset = dataset_factory(
        config["data"]["dataset_factory"],
        (ModelCollectionBuilder(config)
            .with_dataset(dataset_type)
            .filter_category_tags(category_tags)
            .filter_tags(model_tags)
            .random_subset(random_subset)
            .build(dataset_directory)),
    )
    return dataset
