import shutil
from tbparse import SummaryReader
from tbparse.summary_reader import *
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import cv2,os
import numpy as np
from PIL import Image
import glob



class SummaryReaderWithSizes(SummaryReader):
    def __init__(self, log_path: str, *, pivot=False, extra_columns=None, event_types=None,size_guidance=None):
        """The constructor of SummaryReader. Columns contains `step`, `tag`, \
           and `value` by default.

        :param log_path: Load directory location, or load file location.
        :type log_path: str

        :param pivot: Returns long format DataFrame by default, \
                returns wide format DataFrame if set to True. If there are \
                multiple values per step with the same tag, the values are \
                merged into a list.
        :type pivot: bool

        :param extra_columns: Specifies extra columns, defaults to `None`.

                - dir_name:  add a column that contains the relative \
                             directory path.
                - file_name: add a column that contains the relative \
                             event file path.
                - wall_time: add a column that stores the event timestamp.
                - min (histogram): the min value in the histogram.
                - max (histogram): the max value in the histogram.
                - num (histogram): the number of values in the histogram.
                - sum (histogram): the sum of all values in the histogram.
                - sum_squares (histogram): the sum of squares for all values \
                                           in the histogram.
                - width (image): the width of the image.
                - height (image): the height of the image.
                - content_type (audio): the content type of the audio.
                - length_frames (audio): the length of the audio.
                - sample_rate (audio): the sampling rate of the audio.
        :type extra_columns: Set[{'dir_name', 'file_name', 'wall_time', \
                'min', 'max', 'num', 'sum', 'sum_squares', 'width', 'height', \
                'content_type', 'length_frames', 'sample_rate'}]

        :param event_types: Specifies the event types to parse, \
            defaults to all event types.
        :type event_types: Set[{'scalars', 'tensors', 'histograms', 'images', \
            'audio', 'hparams', 'text'}]
        """
        self._log_path: str = log_path
        """Load directory location, or load file location."""
        self._extra_columns: Set[str] = (extra_columns or set()).copy()
        """Specifies additional required columns."""
        if not isinstance(self._extra_columns, set):
            raise ValueError(f"`columns` should be a {set} instead of \
                              {str(type(self._extra_columns))}")
        diff = self._extra_columns - ALL_EXTRA_COLUMNS
        if len(diff) > 0:
            raise KeyError(f"Invalid columns entries: {diff}")
        self._pivot: bool = pivot
        """Determines whether the DataFrame is stored in wide format."""
        self._event_types: Set[str] = (event_types or ALL_EVENT_TYPES).copy()
        """Specifies the event types to parse."""
        if tensorflow is None:
            self._event_types = (event_types or REDUCED_EVENT_TYPES).copy()
        if not isinstance(self._event_types, set):
            raise ValueError(f"`event_types` should be a {set} instead of \
                              {str(type(self._event_types))}")
        diff = self._event_types - ALL_EVENT_TYPES
        if len(diff) > 0:
            raise KeyError(f"Invalid event types: {diff}")
        self._children: Dict[str, 'SummaryReaderWithSizes'] = {}
        """Holds a list of references to the `SummaryReaderWithSizes` children."""

        self._tags: Optional[Dict[str, List[str]]] = None
        """Stores a dictionary contatining a list of parsed tag names for each
        event type."""
        self._events: Dict[str, pd.DataFrame] = self._make_empty_dict(None)
        """Stores a `pandas.DataFrame` containing all events."""

        if not os.path.exists(self.log_path):
            raise ValueError(f"File or directory not found: {self.log_path}")
        if os.path.isfile(self.log_path):
            # Note: tensorflow.python.summary.summary_iterator is less
            #       straightforward, so we use EventAccumulator instead.
            if not size_guidance:
                size_guidance = MINIMUM_SIZE_GUIDANCE.copy()
                for e in self._event_types:
                    size_guidance[e] = 0  # store everything
            event_acc = EventAccumulator(self.log_path, size_guidance)
            event_acc.Reload()
            self._tags = self._make_empty_dict([])
            for e in self._event_types:
                self._parse_events(e, event_acc=event_acc)
        else:
            # Populate children
            for filename in sorted(os.listdir(self.log_path)):
                filepath = os.path.join(self.log_path, filename)
                r = SummaryReaderWithSizes(filepath,
                                  pivot=self._pivot,
                                  extra_columns=self._extra_columns,
                                  event_types=self._event_types)
                self._children[filename] = r



def get_all_log_files(path):
    """
    Recursively search all directories for log files in the given path.
    
    Parameters:
    path (str): The path to list.
    
    Returns:
    list: A list of all log files.
    """
    log_files = []
    dir_list = []
    for root, dirs, files in os.walk(path):
        dir_list.extend(dirs)
        logs = [f for f in files if f.endswith('.0')]
        if len(logs)>0:
            logs = [os.path.join(root,l) for l in logs]
            log_files = log_files+ logs
    return log_files
def save_scalars(log_dir,scalars):
    for tag in scalars:
        if "__None" in tag:
            continue 
        del scalars[tag]['tag']
        scalars[tag].to_csv(f"{os.path.join(log_dir,tag)}.csv",sep=';',header=['step',tag], encoding='utf-8')
    with open(os.path.join(log_dir,'extracted'),'w+') as f:
        f.write("")
def save_images(log_dir,images):
    log_dir = os.path.join(log_dir,'images')
    os.makedirs(log_dir,exist_ok=True)
    for tag in images:
        for step,tag,arr in zip(*list(images[tag].values())):
            Image.fromarray(arr).save(os.path.join(log_dir,f'{tag}_{step}.png'))
            
        
    pass
def extract_info_from_log(log_dir,log_path):
    print(f"extracting scaler and image from -> {log_path}")
    reader = SummaryReaderWithSizes(log_path,size_guidance=size_guidance,pivot=False)
    scalars = reader.scalars
    images = reader.images
    if not images.empty:
        images_dict = {tag_name: (images[images['tag'] == tag_name]).to_dict(orient='list') for tag_name in images.tag.unique()}
        save_images(log_dir,images_dict)
    if not scalars.empty:
        scalars_dict = {tag_name: scalars[scalars['tag'] == tag_name] for tag_name in scalars.tag.unique()}
        save_scalars(log_dir,scalars_dict)
    else:
        print(f"Warning: scalars are empty for log -> {log_dir}")    
    pass
def run_extractor_for_log(log_path):
    log_dir = '/'.join(log_path.split('/')[:-1])
    if REPLACE:
        if os.path.exists(f"{log_dir}/extracted"): os.remove(f"{log_dir}/extracted")
        if os.path.exists(f"{log_dir}/images") :shutil.rmtree(f"{log_dir}/images")
    if not os.path.exists(f"{log_dir}/extracted") :
        extract_info_from_log(log_dir,log_path)

def get_all_images(ea,log_dir):
    images_tag = ea.Tags()['images']
    for it in images_tag:
        events = ea.Images(it)
        for index, event in enumerate(events):
            s = np.frombuffer(event.encoded_image_string, dtype=np.uint8)
            image = cv2.imdecode(s, cv2.IMREAD_COLOR)
            outdir=f"{log_dir}/images"
            os.makedirs(outdir,exist_ok=True)
            filepath = f'{outdir}/{it}_{event.step}_{event.wall_time:.0f}.jpg'
            
            cv2.imwrite(filepath, image)

REPLACE = False
log_dir = 'log_output/unet_cpv5/02_24_2023_23_53_30'
log_path = log_dir+'/events.out.tfevents.1677279210.node070.89304.0'
size_guidance = {'distributions': 1, 'images': 20, 'audio': 0, 'scalars': 0, 'histograms': 0, 'tensors': 0, 'text': 0, 'hparams': 0}
for log_path in get_all_log_files('log_output'):
    run_extractor_for_log(log_path)

# reader = SummaryReaderWithSizes(log_path,size_guidance=size_guidance,pivot=True)

# ea = EventAccumulator(log_path,size_guidance=size_guidance)
# ea.Reload()
# get_all_images(ea,log_dir)
print("")
