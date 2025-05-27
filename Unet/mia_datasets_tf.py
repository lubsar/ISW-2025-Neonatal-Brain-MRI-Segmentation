import tensorflow as tf
import SimpleITK as sitk
import numpy as np
import os
import re

def load_and_process_images(image_paths : list[bytes] | bytes, label_path : bytes) -> tuple[np.ndarray, np.ndarray]:
    image_array = None

    if type(image_paths) is np.ndarray:
        modalities = [sitk.ReadImage(path.decode("utf-8")) for path in image_paths]
        image_array = np.concatenate([np.expand_dims(sitk.GetArrayFromImage(image), axis = -1) for image in modalities], axis = -1)
    else:
       image_array = sitk.GetArrayFromImage(sitk.ReadImage(image_paths.decode("utf-8")))
       
    label_array = sitk.GetArrayFromImage(sitk.ReadImage(label_path.decode("utf-8")))

    return image_array, label_array

def get_image_data(image_path : str) -> tuple:
    img = sitk.ReadImage(image_path)
    array = sitk.GetArrayFromImage(img)

    return img.GetSize()[::-1], img.GetSpacing(), array.dtype

def image_dataset(images_paths : list[tuple[str]], labels_paths : list[str], unified_shape = True) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices((images_paths, labels_paths))

    image_shape, image_spacing, image_datatype = get_image_data(images_paths[0][0] if type(images_paths[0]) is tuple else images_paths[0])
    label_shape, label_spacing, label_datatype = get_image_data(labels_paths[0])

    images = dataset.map(lambda image_paths, label_path: tf.numpy_function(load_and_process_images, [image_paths, label_path], [image_datatype, label_datatype]), num_parallel_calls=tf.data.AUTOTUNE)
    
    if unified_shape != None:
        #fix tensor shape
        def fix_shape(images, label): 
            if type(images_paths[0]) is tuple:
                images.set_shape((*image_shape, len(images_paths[0])))
            else:
                images.set_shape(image_shape)

            label.set_shape(label_shape)

            return images, label

        return images.map(fix_shape, num_parallel_calls=tf.data.AUTOTUNE)

    return images

def convert_to_patches(image : np.ndarray, label : np.ndarray, patch_size : tuple[int], embedings = False):
    if len(image.shape) == 3:
        num_patches = np.ceil(np.array(image.shape) / patch_size).astype(int)
        padding = ((num_patches * patch_size) - image.shape).astype(int)
    else:
        image_shape = image.shape[:-1]
        num_patches = np.ceil(np.array(image_shape) / patch_size).astype(int)
        
        padding = ((num_patches * patch_size) - image_shape).astype(int)
    
    padding_tuple = [(0, pad) for pad in padding]
        
    padded_image = np.pad(image, padding_tuple if len(image.shape) == 3 else (padding_tuple + [(0, 0)]), mode='constant')
    padded_label = np.pad(label, padding_tuple, mode='constant')
        
    patches = []
    label_patches = []

    for idx in np.ndindex(*num_patches):
        patch_selector = tuple(slice(start * patch_size[i], (start + 1) * patch_size[i]) for i, start in enumerate(idx))
        patch = padded_image[patch_selector]
        label_patch = padded_label[patch_selector]

        patches.append(patch)
        label_patches.append(label_patch)
    
    if embedings:
        return np.array(patches), np.array(label_patches), np.linspace(0, len(patches) - 1, len(patches), dtype=int)

    return np.array(patches), np.array(label_patches)

def create_slicer(axis : int):
    def slice_array(img, label):
        img_slice_size = np.full(len(img.shape), -1)
        img_slice_size[axis] = 1

        label_slice_size = np.full(len(label.shape), -1)
        label_slice_size[axis] = 1
       
        num_slices = img.shape[axis]

        img_slices = np.zeros((num_slices, len(img.shape)), dtype=np.int32)
        label_slices = np.zeros((num_slices, len(label.shape)), dtype=np.int32)
        for slice_index in range(num_slices):
            img_slices[slice_index][axis] = slice_index
            label_slices[slice_index][axis] = slice_index

        sliced_dataset = tf.data.Dataset.from_tensor_slices((img_slices, label_slices))

        def operation(img_slices, label_slices):
            image_slice = tf.slice(img, img_slices, img_slice_size)
            label_slice = tf.slice(label, label_slices, label_slice_size)

            if image_slice.shape[axis] == 1:
                image_slice = tf.squeeze(image_slice, axis=axis)

            if label_slice.shape[axis] ==1:
                label_slice = tf.squeeze(label_slice, axis=axis)

            return (image_slice, label_slice)

        sliced_dataset = sliced_dataset.map(operation)
        return sliced_dataset
    
    return slice_array

class CroppedDataset:
    def __init__(self, dataset, min_crop_size=(0, 0, 0)):
        """
        Args:
            dataset: A tf.data.Dataset yielding (image, mask) pairs with shape [D, H, W, C].
            min_crop_size: Tuple (D, H, W) specifying the minimum crop size.
        """
        self.data = dataset
        self.min_crop_size = tf.constant(min_crop_size, dtype=tf.int32)
        self.bbox_min, self.bbox_max = self._compute_bbox()

    def _get_bbox(self, tensor):
        if tf.rank(tensor) == 4:
            tensor = tf.reduce_any(tf.not_equal(tensor, 0), axis=-1)

        nonzero = tf.where(tensor)
        if tf.shape(nonzero)[0] == 0:
            shape = tf.shape(tensor)[:3]
            return tf.zeros(3, dtype=tf.int32), tf.cast(shape, tf.int32)

        min_coords = tf.reduce_min(nonzero, axis=0)
        max_coords = tf.reduce_max(nonzero, axis=0) + 1

        return tf.cast(min_coords, tf.int32), tf.cast(max_coords, tf.int32)

    def _compute_bbox(self):
        bbox_min = None
        bbox_max = None

        for image, mask in self.data.as_numpy_iterator():
            min_mask, max_mask = self._get_bbox(tf.convert_to_tensor(mask))
            min_img, max_img = self._get_bbox(tf.convert_to_tensor(image))

            min_combined = tf.minimum(min_mask, min_img)
            max_combined = tf.maximum(max_mask, max_img)

            if bbox_min is None:
                bbox_min, bbox_max = min_combined, max_combined
            else:
                bbox_min = tf.minimum(bbox_min, min_combined)
                bbox_max = tf.maximum(bbox_max, max_combined)

        return self._expand_bbox_to_min_size(bbox_min, bbox_max)

    def _expand_bbox_to_min_size(self, bbox_min, bbox_max):
        bbox_min = tf.cast(bbox_min, tf.int32)
        bbox_max = tf.cast(bbox_max, tf.int32)

        current_size = bbox_max - bbox_min
        pad_needed = tf.maximum(self.min_crop_size - current_size, 0)

        pad_before = pad_needed // 2
        pad_after = pad_needed - pad_before

        new_min = tf.maximum(bbox_min - pad_before, 0)
        new_max = new_min + tf.maximum(current_size, self.min_crop_size)

        return new_min, new_max

    def crop_fn(self, image, mask):
        def crop_tensor(tensor, target_shape):
            input_shape = tf.shape(tensor)
            rank = tf.rank(tensor)
            size = tf.convert_to_tensor(target_shape, dtype=tf.int32)

            # Centering the crop
            begin = (input_shape - size) // 2
            begin = tf.convert_to_tensor(begin, dtype=tf.int32)  # <-- This is the fix
            leading_dims = tf.cast(rank - tf.shape(target_shape)[0], tf.int32)
            begin = tf.concat([tf.zeros([leading_dims], tf.int32), begin], axis=0)
            size = tf.concat([input_shape[:leading_dims], size], axis=0)

            return tf.slice(tensor, begin, size)

        size = self.bbox_max - self.bbox_min

        return crop_tensor(image, size), crop_tensor(mask, size)

    def dataset(self):
        return self.data.map(lambda x, y: self.crop_fn(x, y))
    
class Iseg2019Original:
    class Subject:
        def __init__(self, number : int, T1_path : str, T2_path : str, label_path : str) -> None:
            self.number = number
            self.T1_path = T1_path
            self.T2_path = T2_path
            self.label_path = label_path

            self.T1 = None
            self.T2 = None
            self.label = None

        def get_T1(self) -> sitk.Image:
            if not self.T1_path:
                return None
            
            if self.T1 is None:
                self.T1 = sitk.ReadImage(self.T1_path)

            return self.T1

        def get_T2(self) -> sitk.Image:
            if not self.T2_path:
                return None
            
            if self.T2 is None:
                self.T2 = sitk.ReadImage(self.T2_path)

            return self.T2

        def get_label(self) -> sitk.Image:
            if not self.label_path:
                return None
            
            if self.label is None:
                self.label = sitk.ReadImage(self.label_path)

            return self.label
        
        def has_T1(self) -> bool:
            return self.T1_path != None
        
        def has_T2(self) -> bool:
            return self.T2_path != None

        def has_label(self) -> bool:
            return self.label_path != None

    def __init__(self, directory : str, extension = "img") -> None:
        if extension.startswith('.'):
            extension = extension[1:]

        subject_number_extractor = lambda string : int(re.search(r"subject-(\d*)-.*", string).group(1))

        subject_numbers = set([subject_number_extractor(file) for file in os.listdir(directory) if file.endswith(extension)])
        self.subjects = []

        for number in subject_numbers:
            T1_path = os.path.join(directory, "subject-{}-T1.{}".format(number, extension))
            T1_path = T1_path if os.path.isfile(T1_path) else None

            T2_path = os.path.join(directory, "subject-{}-T2.{}".format(number, extension))
            T2_path = T2_path if os.path.isfile(T2_path) else None

            label_path = os.path.join(directory, "subject-{}-label.{}".format(number, extension))
            label_path = label_path if os.path.isfile(label_path) else None

            self.subjects.append(Iseg2019Original.Subject(number, T1_path, T2_path, label_path))

    def T1_dataset(self) -> tf.data.Dataset :
        T1_paths = []
        label_paths = []

        for subject in self.subjects:
            T1_paths.append(subject.T1_path)
            label_paths.append(subject.label_path)

        return image_dataset(T1_paths, label_paths)

    def T2_dataset(self) -> tf.data.Dataset:
        T2_paths = []
        label_paths = []

        for subject in self.subjects:
            T2_paths.append(subject.T2_path)
            label_paths.append(subject.label_path)

        return image_dataset(T2_paths, label_paths)

    def T1_T2_dataset(self) -> tf.data.Dataset:
        image_paths = []
        label_paths = []

        for subject in self.subjects:
            image_paths.append((subject.T1_path, subject.T2_path))
            label_paths.append(subject.label_path)

        return image_dataset(image_paths, label_paths)

class Iseg2019Processed(Iseg2019Original):
    def __init__(self, directory : str, extension = "nii.gz") -> None:
        if extension.startswith('.'):
            extension = extension[1:]

        subject_number_extractor = lambda string : int(re.search(r"iseg2019_(\d*).*", string).group(1))

        subject_numbers = set([subject_number_extractor(file) for file in os.listdir(directory) if file.endswith(extension)])
        self.subjects = []

        for number in subject_numbers:
            T1_path = os.path.join(directory, f"iseg2019_{number}_0000.{extension}")
            T1_path = T1_path if os.path.isfile(T1_path) else None

            T2_path = os.path.join(directory, f"iseg2019_{number}_0001.{extension}")
            T2_path = T2_path if os.path.isfile(T2_path) else None

            label_path = os.path.join(directory, f"iseg2019_{number}.{extension}")
            label_path = label_path if os.path.isfile(label_path) else None

            self.subjects.append(Iseg2019Original.Subject(number, T1_path, T2_path, label_path))

class Iseg2017Original(Iseg2019Original):
    def __init__(self, directory : str, extension = "img") -> None:
        super().__init__(directory, extension)

class Iseg2017Processed(Iseg2019Original):
    def __init__(self, directory : str, extension = "nii.gz") -> None:
        if extension.startswith('.'):
            extension = extension[1:]

        subject_number_extractor = lambda string : int(re.search(r"iseg2017_(\d*).*", string).group(1))

        subject_numbers = set([subject_number_extractor(file) for file in os.listdir(directory) if file.endswith(extension)])
        self.subjects = []

        for number in subject_numbers:
            T1_path = os.path.join(directory, f"iseg2017_{number}_0000.{extension}")
            T1_path = T1_path if os.path.isfile(T1_path) else None

            T2_path = os.path.join(directory, f"iseg2017_{number}_0001.{extension}")
            T2_path = T2_path if os.path.isfile(T2_path) else None

            label_path = os.path.join(directory, f"iseg2017_{number}.{extension}")
            label_path = label_path if os.path.isfile(label_path) else None

            self.subjects.append(Iseg2019Original.Subject(number, T1_path, T2_path, label_path))

class BONDID2023Original:
    class Subject:
        def __init__(self, id : str, ADC_ss_path : str, Z_ADC_path : str, label_path : str) -> None:
            self.id = id
            self.ADC_ss_path = ADC_ss_path
            self.Z_ADC_path = Z_ADC_path
            self.label_path = label_path

            self.ADC_ss = None
            self.Z_ADC = None
            self.label = None

        def get_ADC_ss(self) -> sitk.Image:
            if not self.ADC_ss_path:
                return None
            
            if self.ADC_ss is None:
                self.ADC_ss = sitk.ReadImage(self.ADC_ss_path)

            return self.ADC_ss

        def get_Z_ADC(self) -> sitk.Image:
            if not self.Z_ADC_path:
                return None
            
            if self.Z_ADC is None:
                self.Z_ADC = sitk.ReadImage(self.Z_ADC_path)

            return self.Z_ADC

        def get_label(self) -> sitk.Image:
            if not self.label_path:
                return None
            
            if self.label is None:
                self.label = sitk.ReadImage(self.label_path)

            return self.label
        
        def has_ADC_ss(self) -> bool:
            return self.ADC_ss != None
        
        def has_Z_ADC(self) -> bool:
            return self.Z_ADC_path != None

        def has_label(self) -> bool:
            return self.label_path != None

    def __init__(self, directory : str, extension = "mha") -> None:
        if extension.startswith('.'):
            extension = extension[1:]

        adc_ss_directory = os.path.join(directory, "1ADC_ss")
        z_adc_directory = os.path.join(directory, "2Z_ADC")
        label_directory = os.path.join(directory, "3LABEL")

        for  adc_file in os.listdir(adc_ss_directory):
            patient_id = adc_file.split("-")[0]
            adc_path = os.path.join(adc_ss_directory, adc_file)
            z_adc_path = os.path.join(z_adc_directory, f"Zmap_{patient_id}-VISIT_01-ADC_smooth2mm_clipped10.mha")
            label_path = os.path.join(label_directory, f"{patient_id}-VISIT_01_lesion.mha")

            assert(os.path.exists(z_adc_path))
            assert(os.path.exists(label_path))

            self.subjects.append(BONDID2023Original.Subject(patient_id, adc_path, z_adc_path, label_path))

    def ADC_ss_dataset(self) -> tf.data.Dataset :
        ADC_ss_paths = []
        label_paths = []

        for subject in self.subjects:
            ADC_ss_paths.append(subject.ADC_ss_path)
            label_paths.append(subject.label_path)

        return image_dataset(ADC_ss_paths, label_paths)

    def Z_ADC_dataset(self) -> tf.data.Dataset:
        Z_ADC_paths = []
        label_paths = []

        for subject in self.subjects:
            Z_ADC_paths.append(subject.Z_ADC_path)
            label_paths.append(subject.label_path)

        return image_dataset(Z_ADC_paths, label_paths)

    def ADC_ss_Z_ADC_dataset(self) -> tf.data.Dataset:
        image_paths = []
        label_paths = []

        for subject in self.subjects:
            image_paths.append((subject.ADC_ss_path, subject.Z_ADC_path))
            label_paths.append(subject.label_path)

        return image_dataset(image_paths, label_paths)

class BONDID2023Processed(BONDID2023Original):
 def __init__(self, directory : str, extension = "nii.gz") -> None:
        if extension.startswith('.'):
            extension = extension[1:]

        subject_number_extractor = lambda string : int(re.search(r"bonbid_(\d*).*", string).group(1))

        subject_numbers = set([subject_number_extractor(file) for file in os.listdir(directory) if file.endswith(extension)])
        self.subjects : list[BONDID2023Original.Subject] = []

        for number in subject_numbers:
            ADC_ss_path = os.path.join(directory, f"bonbid_{number}_0000.{extension}")
            ADC_ss_path = ADC_ss_path if os.path.isfile(ADC_ss_path) else None

            Z_ADC_path = os.path.join(directory, f"bonbid_{number}_0001.{extension}")
            Z_ADC_path = Z_ADC_path if os.path.isfile(Z_ADC_path) else None

            label_path = os.path.join(directory, f"bonbid_{number}.{extension}")
            label_path = label_path if os.path.isfile(label_path) else None

            self.subjects.append(BONDID2023Original.Subject(number, ADC_ss_path, Z_ADC_path, label_path))

class TACRHIE6Dataset:
    class Subject:
        def __init__(self, number: int, norm_path: str, aseg_path: str, aseg_aparc_path: str) -> None:
            self.number = number
            self.norm_path = norm_path
            self.aseg_path = aseg_path
            self.aseg_aparc_path = aseg_aparc_path

            self.norm = None
            self.aseg = None
            self.aseg_aparc = None

        def get_norm(self) -> sitk.Image:
            if not self.norm_path:
                return None

            if self.norm is None:
                self.norm = sitk.ReadImage(self.norm_path)

            return self.norm

        def get_aseg(self) -> sitk.Image:
            if not self.aseg_path:
                return None

            if self.aseg is None:
                self.aseg = sitk.ReadImage(self.aseg_path)

            return self.aseg

        def get_aseg_aparc(self) -> sitk.Image:
            if not self.aseg_aparc_path:
                return None

            if self.aseg_aparc is None:
                self.aseg_aparc = sitk.ReadImage(self.aseg_aparc_path)

            return self.aseg_aparc

    def __init__(self, directory: str, extension="nii.gz") -> None:
        if extension.startswith('.'):
            extension = extension[1:]

        subject_number_extractor = lambda string: int(re.search(r"sub-(\d*).*", string).group(1))
        subject_numbers = set([subject_number_extractor(file) for file in os.listdir(directory)])
        self.subjects : list[TACRHIE6Dataset.Subject] = [] 

        for number in subject_numbers:
            subject_directory = os.path.join(directory, f"sub-{number:03}")
            
            norm_path = os.path.join(subject_directory, "anat", f"norm.{extension}")
            norm_path = norm_path if os.path.isfile(norm_path) else None

            aseg_path = os.path.join(subject_directory, "segm", f"aseg.{extension}")
            aseg_path = aseg_path if os.path.isfile(aseg_path) else None

            aseg_parc_path = os.path.join(subject_directory, "segm", f"aparc+aseg.{extension}")
            aseg_parc_path = aseg_parc_path if os.path.isfile(aseg_parc_path) else None

            self.subjects.append(TACRHIE6Dataset.Subject(number, norm_path, aseg_path, aseg_parc_path))
            
    def aseg_dataset(self):
        norm_paths = []
        aseg_paths = []

        for subject in self.subjects:
            norm_paths.append(subject.norm_path)
            aseg_paths.append(subject.aseg_path)

        return image_dataset(norm_paths, aseg_paths)

    def aseg_aparc_dataset(self):
        norm_paths = []
        aseg_aparc_paths = []

        for subject in self.subjects:
            norm_paths.append(subject.norm_path)
            aseg_aparc_paths.append(subject.aseg_aparc_path)

        return image_dataset(norm_paths, aseg_aparc_paths)