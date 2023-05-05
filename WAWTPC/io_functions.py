import uproot
import struct
import functools
import sys
import numpy as np

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
    "SimEvent/tracks/tracks.truncatedStartPosUVWT.*",
    "SimEvent/tracks/tracks.truncatedStopPosUVWT.*",
]
################################
################################
def extractData(array, batchSize):
    
    startUVWT = np.array(tuple(zip(*[array[f"tracks.truncatedStartPosUVWT.{direction}"][:] for direction in ["U", "V", "W", "T"]])))
    stopUVWT = np.array(tuple(zip(*[array[f"tracks.truncatedStopPosUVWT.{direction}"][:] for direction in ["U", "V", "W", "T"]])))
    
    iPart=0
    iEvent = 0
    startXYZ = array["tracks.startPos"][iEvent][iPart]
    startXYZ = tvecToArray(startXYZ)
    
    stopXYZ_part0 = array["tracks.stopPos"][iEvent][iPart]
    stopXYZ_part0 = tvecToArray(stopXYZ_part0)
    
    iPart=1
    stopXYZ_part1 = array["tracks.stopPos"][iEvent][iPart]
    stopXYZ_part1 = tvecToArray(stopXYZ_part1)
    
    targetXYZ = np.append(startXYZ, (stopXYZ_part0, stopXYZ_part1))
    targetUVWT = np.append(startUVWT[:,:,iPart][iEvent], 
                           (stopUVWT[:,:,0][iEvent],stopUVWT[:,:,1][iEvent]))
                                               
    projections = array["myChargeArray[3][3][256][512]"]
    projections = np.sum(projections, axis=2)
    projections = np.moveaxis(projections, 1, -1)
     
    return projections[0], targetXYZ 
################################
################################
def extractData_XYZ_UVWT(array, batchSize):
    
    startUVWT = np.array(tuple(zip(*[array[f"tracks.truncatedStartPosUVWT.{direction}"][:] for direction in ["U", "V", "W", "T"]])))
    stopUVWT = np.array(tuple(zip(*[array[f"tracks.truncatedStopPosUVWT.{direction}"][:] for direction in ["U", "V", "W", "T"]])))
    
    iPart = 0
    iEvent = 0
    startXYZ = array["tracks.startPos"][iEvent][iPart]
    startXYZ = tvecToArray(startXYZ)
    
    stopXYZ = array["tracks.stopPos"][iEvent][iPart]
    stopXYZ = tvecToArray(stopXYZ)
             
    return stopXYZ, stopUVWT[0,:,0]
################################
################################
def generator(files):
    batchSize = 1 
    for array in uproot.iterate(files, step_size=batchSize, filter_name=fields, library="np"):
        dataRow = extractData(array, batchSize)
        yield dataRow
################################
################################
def generator_XYZ_UVWT(files):
    batchSize = 1 
    for array in uproot.iterate(files, step_size=batchSize, filter_name=fields, library="np"):
        dataRow = extractData_XYZ_UVWT(array, batchSize)
        yield dataRow
################################
################################ 