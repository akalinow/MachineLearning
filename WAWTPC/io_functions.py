import uproot
import struct
import functools
import sys
import numpy as np
import tensorflow as tf

## Input data shapes
nStrips=256
nTimeSlices = 512
nProj = 3
projections = np.zeros((nStrips,nTimeSlices, nProj))
################################
#slightly modified auto-generated model for tuple, so that it works well as a map key
@functools.total_ordering
class Model_tuple_3c_int_2c_int_2c_int_2c_int_3e__v1(uproot.model.VersionedModel):
    def read_members(self, chunk, cursor, context, file):
        if self.is_memberwise:
            raise NotImplementedError(
                "memberwise serialization of {0}\nin file {1}".format(type(self).__name__, self.file.file_path)
            )
        self._members['_3'], self._members['_2'], self._members['_1'], self._members['_0'] = cursor.fields(chunk, self._format0, context)

    def read_member_n(self, chunk, cursor, context, file, member_index):
        if member_index == 0:
            self._members['_3'] = cursor.field(chunk, self._format_memberwise0, context)
        if member_index == 1:
            self._members['_2'] = cursor.field(chunk, self._format_memberwise1, context)
        if member_index == 2:
            self._members['_1'] = cursor.field(chunk, self._format_memberwise2, context)
        if member_index == 3:
            self._members['_0'] = cursor.field(chunk, self._format_memberwise3, context)
    @classmethod
    def strided_interpretation(cls, file, header=False, tobject_header=True, breadcrumbs=(), original=None):
        if cls in breadcrumbs:
            raise uproot.interpretation.objects.CannotBeStrided('classes that can contain members of the same type cannot be strided because the depth of instances is unbounded')
        breadcrumbs = breadcrumbs + (cls,)
        members = []
        if header:
            members.append(('@num_bytes', np.dtype('>u4')))
            members.append(('@instance_version', np.dtype('>u2')))
        members.append(('_3', np.dtype('>i4')))
        members.append(('_2', np.dtype('>i4')))
        members.append(('_1', np.dtype('>i4')))
        members.append(('_0', np.dtype('>i4')))
        return uproot.interpretation.objects.AsStridedObjects(cls, members, original=original)

    @classmethod
    def awkward_form(cls, file, context):
        from awkward.forms import NumpyForm, ListOffsetForm, RegularForm, RecordForm
        if cls in context['breadcrumbs']:
            raise uproot.interpretation.objects.CannotBeAwkward('classes that can contain members of the same type cannot be Awkward Arrays because the depth of instances is unbounded')
        context['breadcrumbs'] = context['breadcrumbs'] + (cls,)
        contents = {}
        if context['header']:
            contents['@num_bytes'] = uproot._util.awkward_form(np.dtype('u4'), file, context)
            contents['@instance_version'] = uproot._util.awkward_form(np.dtype('u2'), file, context)
        contents['_3'] = uproot._util.awkward_form(np.dtype('>i4'), file, context)
        contents['_2'] = uproot._util.awkward_form(np.dtype('>i4'), file, context)
        contents['_1'] = uproot._util.awkward_form(np.dtype('>i4'), file, context)
        contents['_0'] = uproot._util.awkward_form(np.dtype('>i4'), file, context)
        return RecordForm(contents, parameters={'__record__': 'tuple<int,int,int,int>'})

    _format0 = struct.Struct('>iiii')
    _format_memberwise0 = struct.Struct('>i')
    _format_memberwise1 = struct.Struct('>i')
    _format_memberwise2 = struct.Struct('>i')
    _format_memberwise3 = struct.Struct('>i')
    base_names_versions = []
    member_names = ['_3', '_2', '_1', '_0']
    class_flags = {}

    def postprocess(self, chunk, cursor, context, file):
        return tuple(self._members.values())


    def __eq__(self, other):
        return self._members == other._members

    def __lt__(self, other):
        if self._members["_0"]<other.members["_0"]:
            if self._members["_1"]<other.members["_1"]:
                if self._members["_2"]<other.members["_2"]:
                    if self._members["_3"]<other.members["_3"]:
                        return True
        return False
################################
################################
def tvecToArray(v):
	return np.array([v.member("fX"), v.member("fY"), v.member("fZ")])
################################
################################
fields = [
    #"SimEvent/reactionType",
    "SimEvent/tracks/tracks.startPos",
    "SimEvent/tracks/tracks.stopPos",
    #"SimEvent/tracks/tracks.prim.pID",
    #"SimEvent/tracks/tracks.prim.fourMomentum",
    #"Event/myChargeMap",
    "Event/myChargeArray*",
    #"SimEvent/tracks/tracks.truncatedStartPosUVWT.*",
    #"SimEvent/tracks/tracks.truncatedStopPosUVWT.*",
]
################################
################################
def generator(files, batchSize, features_only=False):
    for array in uproot.iterate(files, step_size=batchSize, 
                                filter_name=fields, 
                                num_workers = 4, 
                                library="ak"):
        
        features = array["myChargeArray[3][3][256][512]"].to_numpy()
        features = features.astype(float)
        features = np.sum(features, axis=2)
        features = np.moveaxis(features, 1, -1)
        features /= np.amax(features, axis=(1,2,3), keepdims=True)
        features = (features>0.05)*features
        
        if features_only:
            yield features, np.full((batchSize, 9), 0.0) 
        else:    
            fX = array['tracks.startPos']['fX'].to_numpy()
            fY = array['tracks.startPos']['fY'].to_numpy()
            fZ = array['tracks.startPos']['fZ'].to_numpy()
            startPos = np.stack([fX, fY, fZ], axis=1)[:,:,[0]]

            fX = array['tracks.stopPos']['fX'].to_numpy()
            fY = array['tracks.stopPos']['fY'].to_numpy()
            fZ = array['tracks.stopPos']['fZ'].to_numpy()
            stopPos = np.stack([fX, fY, fZ], axis=1)

            target = np.concatenate([startPos, stopPos], axis=2)
            target /= 100.0

            yield features, target.reshape(batchSize, -1, order='F')
################################
################################