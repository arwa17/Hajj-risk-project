import json
import unittest
from types import SimpleNamespace

from hajj_study.generalist import execute_single_case
from hajj_study.gate import SharedDecisionGate
from hajj_study.verify import VerificationOracle
from hajj_study.provider import ProviderError, CallFailure

CID='HJ-V004'
CLEAN={'Crowd_Density':'Low','Temperature':39.,'Fatigue_Level':'High',
       'Stress_Level':'Medium','Queue_Time_minutes':50.,'Emergency_Event':'No',
       'Distance_Between_People_m':1.2}
PRED={'predicted_risk':'MEDIUM','probabilities':{'LOW':.2,'MEDIUM':.6,'HIGH':.2}}


def report(flags,risk='HIGH'):
    return {'case_id':CID,'proposed_risk_level':risk,'evidence_flags':flags,
            'supporting_evidence':['Observed evidence supports this proposed risk.']}


def flag(field,kind):
    return {'field':field,'type':kind,'reason':'Uncertain observation requires review.'}

class FakeProvider:
    def __init__(self,answers):self.answers=list(answers);self.prompts=[]
    def generate_validated(self,*,system_prompt,user_prompt,response_schema,validator):
        self.prompts.append(json.loads(user_prompt))
        obj=self.answers.pop(0)
        if isinstance(obj,Exception):raise obj
        obj=validator(obj)
        return SimpleNamespace(report=obj,attempts=[{'status':'STOP','attempt':1,
                              'usage':{'total_tokens':73,'thoughts_tokens':8}}])

class GeneralistTests(unittest.TestCase):
    def case(self,corrupt=False):
        obs=dict(CLEAN)
        if corrupt:
            obs['Crowd_Density']=None
            obs['Temperature']=None
        return {'case_id':CID,'observations':obs,'ml_prediction':PRED}

    def oracle(self):return VerificationOracle({CID:CLEAN})

    def test_issue_without_verification(self):
        provider=FakeProvider([report([])])
        res=execute_single_case(self.case(),provider=provider,gate=SharedDecisionGate(7),oracle=self.oracle())
        self.assertEqual((res.status,res.risk_level,res.trace['verification_count']),('ISSUE','HIGH',0))
        self.assertEqual(res.trace['logical_agent_calls'],1)
        self.assertEqual(len(res.trace['api_attempts']),1)

    def test_verify_once_then_abstain_after_inheriting_other_flags(self):
        before=report([flag('Crowd_Density','conflicting'),flag('Temperature','missing')])
        after=report([],risk='LOW')
        provider=FakeProvider([before,after])
        res=execute_single_case(self.case(corrupt=True),provider=provider,
                                gate=SharedDecisionGate(7),oracle=self.oracle())
        self.assertEqual((res.status,res.risk_level),('ABSTAIN',None))
        self.assertEqual(res.trace['verification_domain'],'crowd')
        self.assertEqual(res.trace['verification_count'],1)
        self.assertEqual(res.trace['logical_agent_calls'],2)
        self.assertEqual(res.trace['post_gate']['merged_flags'][0]['field'],'Temperature')
        self.assertEqual(provider.prompts[1]['observations']['Crowd_Density'],'Low')
        self.assertIsNone(provider.prompts[1]['observations']['Temperature'])
        self.assertEqual(provider.prompts[1]['verified_domain'],'crowd')

    def test_verify_then_issue(self):
        before=report([flag('Temperature','missing')])
        provider=FakeProvider([before,report([],risk='MEDIUM')])
        res=execute_single_case(self.case(corrupt=True),provider=provider,
                                gate=SharedDecisionGate(7),oracle=self.oracle())
        self.assertEqual((res.status,res.risk_level,res.trace['verification_count']),('ISSUE','MEDIUM',1))

    def test_provider_failure_is_failed_not_abstain(self):
        err=ProviderError('invalid provider response',category='MAX_TOKENS',finish_reason='MAX_TOKENS')
        failed=CallFailure(err,[{'attempt':1,'status':'MAX_TOKENS'}])
        res=execute_single_case(self.case(),provider=FakeProvider([failed]),
                                gate=SharedDecisionGate(7),oracle=self.oracle())
        self.assertEqual(res.status,'FAILED')
        self.assertIsNone(res.risk_level)
        self.assertEqual(res.trace['failure']['category'],'MAX_TOKENS')
        self.assertEqual(len(res.trace['api_attempts']),1)

if __name__=='__main__':unittest.main()
