import SimpleITK as sitk
import numpy as np
import os
import matplotlib.pyplot as plt
import nibabel as nib
from nilearn import plotting

def plotVoxels(array : np.ndarray) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.voxels(array)
    ax.set_box_aspect([1, 1, 1])

    return fig

def writeImage(image : sitk.Image, path : str) -> None:
    sitk.WriteImage(image, path)

def readImage(path : str) -> sitk.Image:
    return sitk.ReadImage(path)

def writeImageArray(array : np.ndarray, path : str, metadata_img) -> None:
    if(len(array.shape) != 3):
        raise ValueError(f"Array has unsupported dimension: {array.shape}")
    
    image = sitk.GetImageFromArray(array)
    image.CopyInformation(metadata_img)
    sitk.WriteImage(image, path)

def writeImagesArray(array : np.ndarray, directory : str, name_generator, metadata_img_getter) -> None:
    if(len(array.shape) > 4 or len(array.shape) < 3):
        raise ValueError(f"Array has unsupported dimension: {array.shape}")
    
    if len(array.shape) == 4:
        for image_index in range(array.shape[0]):
            name = name_generator(image_index)
            metadata_img = metadata_img_getter(image_index)

            writeImageArray(array[image_index], os.path.join(directory, name), metadata_img)
    else:
        writeImageArray(array, os.path.join(directory, name_generator(0)), metadata_img_getter(0))
    

def resample_image(image, new_size, interpolator : sitk.sitkLinear):
    for dimension in new_size:
        if not isinstance(dimension, int):
            raise TypeError(f"Dimension type of size must be int not {type(dimension)}")
    
    dimension = image.GetDimension()
    reference_physical_size = np.zeros(image.GetDimension())
    reference_physical_size[:] = [(sz-1) * spc if sz * spc > mx  else mx for sz, spc, mx in zip(image.GetSize(), image.GetSpacing(), reference_physical_size)]
    
    reference_origin = image.GetOrigin()
    reference_direction = image.GetDirection()

    reference_size = new_size
    reference_spacing = [ phys_sz / (sz-1) for sz, phys_sz in zip(reference_size, reference_physical_size) ]

    reference_image = sitk.Image(reference_size, image.GetPixelIDValue())
    reference_image.SetOrigin(reference_origin)
    reference_image.SetSpacing(reference_spacing)
    reference_image.SetDirection(reference_direction)

    reference_center = np.array(reference_image.TransformContinuousIndexToPhysicalPoint(np.array(reference_image.GetSize())/2.0))
    
    transform = sitk.AffineTransform(dimension)
    transform.SetMatrix(image.GetDirection())
   
    transform.SetTranslation(np.array(image.GetOrigin()) - reference_origin)
  
    centering_transform = sitk.TranslationTransform(dimension)
    img_center = np.array(image.TransformContinuousIndexToPhysicalPoint(np.array(image.GetSize())/2.0))
    centering_transform.SetOffset(np.array(transform.GetInverse().TransformPoint(img_center) - reference_center))
    centered_transform = sitk.CompositeTransform(transform)
    centered_transform.AddTransform(centering_transform)

    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(interpolator)
    resampler.SetTransform(centered_transform)
    resampler.SetSize(new_size)
    resampler.SetOutputSpacing(reference_spacing)
    resampler.SetOutputOrigin(reference_origin)
    resampled_image = resampler.Execute(image)

    return resampled_image

def resize_image_keep_spacing(image, new_size, interpolator = sitk.sitkLinear):
    for dimension in new_size:
        if not isinstance(dimension, int):
            raise TypeError(f"Dimension type of size must be int not {type(dimension)}")
        
    dimension = image.GetDimension()
    reference_physical_size = np.zeros(image.GetDimension())
    reference_physical_size[:] = [(sz-1) * spc if sz * spc > mx  else mx for sz, spc, mx in zip(image.GetSize(), image.GetSpacing(), reference_physical_size)]
    
    reference_origin = image.GetOrigin()
    reference_direction = image.GetDirection()

    reference_size = new_size
    reference_spacing = image.GetSpacing()

    reference_image = sitk.Image(reference_size, image.GetPixelIDValue())
    reference_image.SetOrigin(reference_origin)
    reference_image.SetSpacing(reference_spacing)
    reference_image.SetDirection(reference_direction)

    reference_center = np.array(reference_image.TransformContinuousIndexToPhysicalPoint(np.array(reference_image.GetSize())/2.0))
    
    transform = sitk.AffineTransform(dimension)
    transform.SetMatrix(image.GetDirection())
    transform.SetCenter(reference_center)
    transform.Scale([ phys_sz / (sz-1) for sz,phys_sz in zip(reference_size, reference_physical_size) ])
  
    centering_transform = sitk.TranslationTransform(dimension)
    img_center = np.array(image.TransformContinuousIndexToPhysicalPoint(np.array(image.GetSize())/2.0))
    centering_transform.SetOffset(np.array(transform.GetInverse().TransformPoint(img_center) - reference_center))
    centered_transform = sitk.CompositeTransform(transform)
    centered_transform.AddTransform(centering_transform)

    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(interpolator)
    resampler.SetTransform(centered_transform)
    resampler.SetSize(new_size)
    resampler.SetOutputSpacing(reference_spacing)
    resampler.SetOutputOrigin(reference_origin)
    resampled_image = resampler.Execute(image)

    return resampled_image

def resize_image(image : sitk.Image, new_size : tuple[int, int, int], new_spacing : tuple[float, float, float], interpolator = sitk.sitkLinear) ->sitk.Image:
    for dimension in new_size:
        if not isinstance(dimension, int):
            raise TypeError(f"Dimension type of size must be int not {type(dimension)}")
        
    for dimension in new_spacing:
        if not isinstance(dimension, float):
            raise TypeError(f"Dimension type of spacing must be float not {type(dimension)}")
    
    dimension = image.GetDimension()
    image_physical_size = np.zeros(image.GetDimension())
    image_physical_size[:] = [(sz-1) * spc if sz * spc > mx  else mx for sz, spc, mx in zip(image.GetSize(), image.GetSpacing(), image_physical_size)]
    
    reference_direction = image.GetDirection()

    reference_size = new_size
    reference_physical_size = np.zeros(image.GetDimension())
    reference_physical_size[:] = [(sz-1) * spc if sz * spc > mx  else mx for sz, spc, mx in zip(new_size, new_spacing, reference_physical_size)]
   
    reference_spacing = new_spacing
    reference_image = sitk.Image(reference_size, image.GetPixelIDValue())
    reference_origin = np.array(reference_size) / 2.0
    print(image.GetOrigin(), reference_origin)
    
    reference_image.SetOrigin(reference_origin)
    reference_image.SetSpacing(reference_spacing)
    reference_image.SetDirection(reference_direction)

    reference_center = np.array(reference_image.TransformContinuousIndexToPhysicalPoint(np.array(reference_image.GetSize())/2.0))
    
    transform = sitk.AffineTransform(dimension)
    transform.SetMatrix(image.GetDirection())
    transform.Scale([ phys_sz / (sz-1) for sz, phys_sz in zip(reference_physical_size, image_physical_size) ])
    transform.SetTranslation(np.array(image.GetOrigin()) - reference_origin)
  
    centering_transform = sitk.TranslationTransform(dimension)
    img_center = np.array(image.TransformContinuousIndexToPhysicalPoint(np.array(image.GetSize())/2.0))
    centering_transform.SetOffset(np.array(transform.GetInverse().TransformPoint(img_center) - reference_center))
    centered_transform = sitk.CompositeTransform(transform)
    centered_transform.AddTransform(centering_transform)

    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(interpolator)
    resampler.SetTransform(centered_transform)
    resampler.SetSize(new_size)
    resampler.SetOutputSpacing(reference_spacing)
    resampler.SetOutputOrigin(reference_origin)
    resampled_image = resampler.Execute(image)

    return resampled_image

def interactive_display(array : np.ndarray, value_range : tuple[int, int],
                         title = None, background : np.ndarray | None = None, cmap = None):
    nifti_img = nib.Nifti1Image(array, affine=np.eye(4), dtype=np.uint8)

    display = plotting.view_img(nifti_img, bg_img=background, colorbar=False, symmetric_cmap=False,
                                 cmap=cmap, black_bg=True, vmin=value_range[0], vmax=value_range[1], title=title)
    display.resize(1000, 250)

    return display

def embed_tensor(small_tensor, target_shape, insert_at):
    output = np.zeros(target_shape, dtype=small_tensor.dtype)

    slices = tuple(slice(insert_at[i], insert_at[i] + small_tensor.shape[i]) for i in range(len(insert_at)))
    
    output[slices] = small_tensor
    return output