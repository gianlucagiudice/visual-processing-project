import cv2
import numpy as np
from tqdm import tqdm


class MetadataUtils:

    def __init__(self):
        pass

    def reorder_columns(self, dataset, head):
        columns = dataset.columns.tolist()
        tail = set(columns) - set(head)
        ordered_columns = head + list(tail)
        # Ordered columns
        df = dataset[ordered_columns]

        return df

    def remove_invalid_rows(self, dataset):
        len_before = len(dataset)
        print('Len before: ', len_before)
        dataset = dataset.query('age<=100')
        dataset = dataset[dataset.gender.notna()]
        dataset = dataset[dataset.age.notna()]
        len_after = len(dataset)
        print('Len after: ', len_after)
        print(f'Invalid rows: {(1 - len_after / len_before) * 100:.3f}%')

        return dataset

    def are_rows_equal(self, rows):
        check = False
        for i in range(1, len(rows)):
            if (rows[i - 1] == rows[i]).all():
                check = True
            else:
                check = False
        return check

    def are_cols_equal(self, cols):
        return self.are_rows_equal(np.moveaxis(cols, 0, 1))

    def is_image_padded(self, img, number_equal_rows=5):
        nr = number_equal_rows
        return self.are_rows_equal(img[:nr, :, :]) or self.are_rows_equal(img[-nr:, :, :]) or \
               self.are_cols_equal(img[:, :nr, :]) or self.are_cols_equal(img[:, -nr:, :])

    def is_image_too_little(self, img, smallest_dim=124):
        if img.shape[0] <= smallest_dim or img.shape[1] <= smallest_dim:
            return True
        else:
            return False

    def remove_invalid_images(self, dataset, path):
        len_before = len(dataset)
        print('Len before: ', len_before)

        with tqdm(total=dataset.shape[0]) as pbar:
            for index, row in dataset.iterrows():
                img = cv2.imread(path + row.full_path)
                if self.is_image_too_little(img) or self.is_image_padded(img):
                    dataset.drop(index, inplace=True)
                pbar.update(1)

        len_after = len(dataset)
        print('Len after: ', len_after)
        print(f'Invalid rows: {(1 - len_after / len_before) * 100:.3f}%')

        return dataset
