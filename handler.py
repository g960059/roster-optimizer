import pulp
import numpy as np
from box import Box
import json
import copy

def solve(req):
    req = Box(req)
    req['groups'] = [g for g in req['groups'] if len(g['members']['items']) != 0]
    N_shiftTypes = len(req.get('shiftTypes'))
    N_dates = len(req.get('dates'))
    N_members = len(req.get('members'))
    N_groups = len(req.get('groups'))
    shiftTypeIndexToId = [m['id'] for m in req['shiftTypes']]
    shiftTypeIdToIndex = {id: idx for idx,id in enumerate(shiftTypeIndexToId)}
    memberIndexToId = [m['id'] for m in req['members']]
    memberIdToIndex = {id: idx for idx,id in enumerate(memberIndexToId)}
    groupIndexToId = [g['id'] for g in req['groups']]
    groupIdToIndex = {id: idx for idx,id in enumerate(groupIndexToId)}
    dateFromIndex = req['dates']
    dateToIndex = {id: idx for idx,id in enumerate(dateFromIndex)}
    groupIndexToMembers = [[memberIdToIndex[m['intervalMemberID']] for m in g['members']['items']] for g in req['groups']]
    lp = pulp.LpProblem()
    x = np.array([pulp.LpVariable(name='x_member({})_date({})_shiftType({})'.format(m,d,w),cat='Binary') for m in range(N_members) for d in range(N_dates) for w in range(N_shiftTypes)]).reshape((N_members,N_dates,N_shiftTypes))
    minRequirementDiff = np.array([pulp.LpVariable(name='minRequirementDiff_group({})_date({})_shiftType({})'.format(g,d,w), lowBound=0) for g in range(N_groups) for d in range(N_dates) for w in range(N_shiftTypes)]).reshape(N_groups,N_dates,N_shiftTypes)
    maxRequirementDiff =  np.array([pulp.LpVariable(name='maxRequirementDiff_group({})_date({})_shiftType({})'.format(g,d,w), lowBound=0) for g in range(N_groups) for d in range(N_dates) for w in range(N_shiftTypes)]).reshape(N_groups,N_dates,N_shiftTypes)
    minTotalCountDiff = np.array([pulp.LpVariable(name='minTotalCountDiff_member({})_shiftType({})'.format(m,w), lowBound=0) for m in range(N_members) for w in range(N_shiftTypes)]).reshape(N_members,N_shiftTypes)
    maxTotalCountDiff = np.array([pulp.LpVariable(name='maxTotalCountDiff_member({})_shiftType({})'.format(m,w), lowBound=0) for m in range(N_members) for w in range(N_shiftTypes)]).reshape(N_members,N_shiftTypes)
    minConsecutiveDiff = np.array([pulp.LpVariable(name='minConsecutiveDiff_member({})_shiftType({})'.format(m,w), lowBound=0) for m in range(N_members) for w in range(N_shiftTypes)]).reshape(N_members,N_shiftTypes)
    maxConsecutiveDiff = np.array([pulp.LpVariable(name='maxConsecutiveDiff_member({})_shiftType({})'.format(m,w), lowBound=0) for m in range(N_members) for w in range(N_shiftTypes)]).reshape(N_members,N_shiftTypes)
    minIntervalDiff = np.array([pulp.LpVariable(name='minIntervalDiff_member({})_shiftType({})'.format(m,w), lowBound=0) for m in range(N_members) for w in range(N_shiftTypes)]).reshape(N_members,N_shiftTypes)
    maxIntervalDiff = np.array([pulp.LpVariable(name='maxIntervalDiff_member({})_shiftType({})'.format(m,w), lowBound=0) for m in range(N_members) for w in range(N_shiftTypes)]).reshape(N_members,N_shiftTypes)
    # prohibitedPatternDiff = np.array([pulp.LpVariable(name='prohibitedPatternDiff_member({})_shiftType({})'.format(m,w), lowBound=0) for m in range(N_members) for w in range(N_shiftTypes)]).reshape(N_members,N_shiftTypes)
    requestObjectives = []

    # collision
    for m in range(N_members):
        for d in range(N_dates):
            lp += pulp.lpSum(x[m,d]) <= 1, 'collision_member{}_date{}'.format(m,d)
    # Requirements
    for r in req['requirements']:
        groupIndex = groupIdToIndex[r['intervalGroupID']]
        shiftTypeIndex = shiftTypeIdToIndex[r['intervalShiftTypeID']]
        memberIndices = groupIndexToMembers[groupIndex]
        for d in r['dates']['items']:
            dateIndex = dateToIndex[d['date']]
            lp += d['min'] - minRequirementDiff[groupIndex,dateIndex,shiftTypeIndex] <= pulp.lpSum(x[memberIndices,dateIndex,shiftTypeIndex]),'minRequirement_group{}_date{}_shiftType{}'.format(groupIndex,dateIndex,shiftTypeIndex)
            lp += d['max'] + maxRequirementDiff[groupIndex,dateIndex,shiftTypeIndex] >= pulp.lpSum(x[memberIndices,dateIndex,shiftTypeIndex]),'maxRequirement_group{}_date{}_shiftType{}'.format(groupIndex,dateIndex,shiftTypeIndex)

    
    # Member Constraints
    for memberIndex,member in enumerate(req.members):
        if member.constraintSet is None:
            continue
        # Total Counts
        for item in member.constraintSet.totalCountRanges['items']:
            shiftTypeIndex = shiftTypeIdToIndex[item.intervalShiftTypeID]
            totalCount = pulp.lpSum(x[memberIndex,:,shiftTypeIndex])
            if item.min is not None:
                lp +=  item.min - minTotalCountDiff[memberIndex,shiftTypeIndex] <= totalCount, "minTotalCount_member{}_shiftType{}".format(memberIndex,shiftTypeIndex)
            if item.max is not None:
                lp +=  totalCount  <= item.max + maxTotalCountDiff[memberIndex,shiftTypeIndex], "maxTotalCount_member{}_shiftType{}".format(memberIndex,shiftTypeIndex)

        # Consecutive 
        for item in member.constraintSet.consecutiveRanges['items']:
            shiftTypeIndex = shiftTypeIdToIndex[item.intervalShiftTypeID]
            if item.max is not None:
                for dateIndex in range(item.max,N_dates):
                    lp += pulp.lpSum(x[memberIndex,dateIndex-dateDiff,shiftTypeIndex] for dateDiff in range(item.max+1)) - maxConsecutiveDiff[memberIndex,shiftTypeIndex] <= item.max
            if item.min is not None:
                for dateIndex in range(item.min, N_dates):
                    lp += pulp.lpSum([x[memberIndex,dateIndex-dateDiff,shiftTypeIndex] for dateDiff in range(2,item.min+1)]) - (item.min-1) * x[memberIndex,dateIndex-1,shiftTypeIndex] + (item.min-1) * x[memberIndex,dateIndex,shiftTypeIndex] + minConsecutiveDiff[memberIndex,shiftTypeIndex] >= 0
        # Interval
        for item in member.constraintSet.intervalRanges['items']:
            shiftTypeIndex = shiftTypeIdToIndex[item.intervalShiftTypeID]
            if item.max is not None:
                for dateIndex in range(item.max, N_dates):
                    lp += pulp.lpSum(x[memberIndex,dateIndex-dateDiff,shiftTypeIndex] for dateDiff in range(item.max+1)) + maxIntervalDiff[memberIndex,shiftTypeIndex] >= 1
            if item.min is not None:
                for dateIndex in range(item.min, N_dates):
                    for t in range(2,item.min+1):
                        lp += x[memberIndex,dateIndex-t,shiftTypeIndex] - pulp.lpSum(x[memberIndex,dateIndex-dateDiff,shiftTypeIndex] for dateDiff in range(1,t)) + x[memberIndex,dateIndex,shiftTypeIndex] - minIntervalDiff[memberIndex,shiftTypeIndex] <= 1
        
        # Prohibited Pattern preprocessing
        prohibitedPatterns = []
        for patternItem in member.constraintSet.prohibitedPatterns['items']:
            patternList = [[]]
            for item in patternItem.pattern['items']:
                tmpPatternList = copy.deepcopy(patternList)
                newPattern = []
                for ptn in tmpPatternList:
                    # print('ptn:', ptn)
                    for i in item.prohibitedShiftTypes['items']:
                        tmpPtn = copy.deepcopy(ptn)
                        tmpPtn.append(i.intervalShiftTypeID)
                        newPattern.append(tmpPtn)
                # print('newPattern: ', len(newPattern))
                patternList = newPattern
            prohibitedPatterns.extend(patternList)

        # Prohibited Pattern
        for pattern in prohibitedPatterns:
            L = len(pattern)
            for dateIndex in range(L-1, N_dates):
                lp += pulp.lpSum(x[memberIndex,dateIndex-L+1+dateDiff,shiftTypeIdToIndex[pattern[dateDiff]]] for dateDiff in range(L))  <= L-1

        # Fixed Assignment
        for item in member.fixedAssignments['items']:
            shiftTypeIndex = shiftTypeIdToIndex[item.intervalShiftTypeID]
            dateIndex = dateToIndex[item.date]
            shiftTypeIndexExcluded = list(range(N_shiftTypes))
            shiftTypeIndexExcluded.remove(shiftTypeIndex)
            lp += x[memberIndex,dateIndex,shiftTypeIndex] == 1
            for idx in shiftTypeIndexExcluded:
                lp += x[memberIndex,dateIndex,idx] == 0
        # Requests 
        requestObjectives.append(pulp.lpSum(1-x[memberIndex,dateToIndex[item.date],shiftTypeIdToIndex[item.intervalShiftTypeID]] for item in member.requests['items']))
            


    # Objective Function
    lp += pulp.lpSum([minRequirementDiff,maxRequirementDiff, minTotalCountDiff,maxTotalCountDiff,minConsecutiveDiff,maxConsecutiveDiff,maxIntervalDiff,minIntervalDiff,requestObjectives])
    lp.solve()

    print(pulp.LpStatus[lp.status])
    print('objective: ', pulp.value(lp.objective))
    print(np.vectorize(lambda x: x.value())(x))

    def formatAssignment(assignment):
        res = []
        for memberIdx in range(N_members):
            dates = []
            for dateIdx in range(N_dates):
                dates.append({'date': dateFromIndex[dateIdx], 'intervalShiftTypeID': shiftTypeIndexToId[(assignment[memberIdx][dateIdx]@np.arange(N_shiftTypes)).astype(int)]})
            member = {'intervalMemberID': memberIndexToId[memberIdx], 'dates':dates}
            res.append(member)
        return res
    return formatAssignment(np.vectorize(lambda x: x.value())(x))


def optimize(event, context):
    req = json.loads(event['body'])
    res = solve(req)
    response = {
        "statusCode":200,
        "headers": {
            "Content-Type": 'application/json',
            "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
            "Access-Control-Allow-Methods": "POST",
            "Access-Control-Allow-Origin": "*"
        },
        "body": json.dumps(res)
    }
    return response

    # Use this code if you don't use the http event with the LAMBDA-PROXY
    # integration
    """
    return {
        "message": "Go Serverless v1.0! Your function executed successfully!",
        "event": event
    }
    """
