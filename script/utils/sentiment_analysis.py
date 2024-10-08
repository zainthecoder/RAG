from pyabsa import (
    AspectTermExtraction as ATEPC,
    DeviceTypeOption,
    available_checkpoints,
)
from pyabsa import TaskCodeOption

checkpoint_map = available_checkpoints(
    TaskCodeOption.Aspect_Polarity_Classification, show_ckpts=True
)
# checkpoint_map = available_checkpoints()


aspect_extractor = ATEPC.AspectExtractor('english', auto_device=DeviceTypeOption.AUTO)
# aspect_extractor = ATEPC.AspectExtractor('chinese', auto_device=DeviceTypeOption.AUTO)

atepc_result=aspect_extractor.predict(['the phone battery is fine, camera is horrible and display is fine'],
                         save_result=True,
                         print_result=True,  # print the result
                         ignore_error=True,  # ignore the error when the model cannot predict the input
                         )
print(atepc_result)                                                           