	1�Z$$@1�Z$$@!1�Z$$@	RQ�B��?RQ�B��?!RQ�B��?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$1�Z$$@�A`��"�?AJ+��#@Y�p=
ף�?*	      W@2F
Iterator::Model��~j�t�?!zӛ���D@)���S㥛?1�,d!Y=@:Preprocessing2j
3Iterator::Model::ParallelMap::Zip[1]::ForeverRepeaty�&1��?!�7��Mo>@)9��v���?1d!Y�B<@:Preprocessing2d
-Iterator::Model::ParallelMap::Zip[0]::FlatMap;�O��n�?!Y�B��3@)�~j�t��?1!Y�B*@:Preprocessing2S
Iterator::Model::ParallelMap�I+��?!���7��'@)�I+��?1���7��'@:Preprocessing2X
!Iterator::Model::ParallelMap::Zip���S㥫?!�,d!YM@)����Mb�?1���,d!@:Preprocessing2t
=Iterator::Model::ParallelMap::Zip[0]::FlatMap[0]::TensorSlice�~j�t�x?!!Y�B@)�~j�t�x?1!Y�B@:Preprocessing2v
?Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat::FromTensor����Mb`?!���,d@)����Mb`?1���,d@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.6% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�A`��"�?�A`��"�?!�A`��"�?      ��!       "      ��!       *      ��!       2	J+��#@J+��#@!J+��#@:      ��!       B      ��!       J	�p=
ף�?�p=
ף�?!�p=
ף�?R      ��!       Z	�p=
ף�?�p=
ף�?!�p=
ף�?JCPU_ONLY