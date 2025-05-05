import argparse
from dataclasses import dataclass, field, asdict
import json


@dataclass
class DisaggregationMethod:
    n_enode: int
    n_pnode: int
    n_dnode: int
    n_epnode: int
    n_ednode: int
    n_pdnode: int
    n_epdnode: int


@dataclass
class DisaggregationMethods:
    epd  : dict[int, list[DisaggregationMethod]] = field(default_factory=dict)
    ep_d : dict[int, list[DisaggregationMethod]] = field(default_factory=dict)
    ed_p : dict[int, list[DisaggregationMethod]] = field(default_factory=dict)
    e_p_d: dict[int, list[DisaggregationMethod]] = field(default_factory=dict)


def search_disaggregation_methods(max_n_instance: int):
    methods = DisaggregationMethods()
    for n_instance in range(1, max_n_instance + 1):
        methods.epd[n_instance] = []
        methods.ep_d[n_instance] = []
        methods.ed_p[n_instance] = []
        methods.e_p_d[n_instance] = []
        for e in range(0, n_instance + 1):
            for ep in range(0, n_instance + 1):
                for ed in range(0, n_instance + 1):
                    for epd in range(0, n_instance + 1):
                        for p in range(0, n_instance + 1):
                            for pd in range(0, n_instance + 1):
                                for d in range(0, n_instance + 1):
                                    method = DisaggregationMethod(n_enode = e, n_pnode = p, n_dnode = d, n_ednode = ed, n_epnode = ep, n_pdnode = pd, n_epdnode = epd)
                                    if (e + ep + ed + epd + p + pd + d) != n_instance:
                                        continue

                                    has_e = e > 0 or ep > 0 or ed > 0 or epd > 0
                                    has_p = ep > 0 or epd > 0 or p > 0 or pd > 0
                                    has_d = ed > 0 or epd > 0 or pd > 0 or d > 0
                                    if not has_e or not has_p or not has_d:
                                        continue

                                    if (
                                        e == 0 and
                                        ep == 0 and
                                        ed == 0 and
                                        epd != 0 and
                                        p == 0 and
                                        pd == 0 and
                                        d == 0
                                    ):
                                        methods.epd[n_instance].append(method)

                                    if (
                                        e != 0 and
                                        ep == 0 and
                                        ed == 0 and 
                                        epd == 0 and
                                        p != 0 and
                                        pd == 0 and
                                        d != 0
                                    ): 
                                        methods.e_p_d[n_instance].append(method)

                                    if (
                                        e == 0 and
                                        ep != 0 and
                                        ed == 0 and
                                        epd == 0 and
                                        p == 0 and
                                        pd == 0 and
                                        d != 0
                                    ): 
                                        methods.ep_d[n_instance].append(method)

                                    if (
                                        e == 0 and
                                        ep == 0 and
                                        ed != 0 and
                                        epd == 0 and
                                        p != 0 and
                                        pd == 0 and
                                        d == 0
                                    ): 
                                        methods.ed_p[n_instance].append(method)

    return methods

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-instance', type=int, default=1, help='Number of instances')
    args = parser.parse_args()
    n_instance = args.n_instance
    methods = search_disaggregation_methods(n_instance)

    print(json.dumps(asdict(methods), indent=4))