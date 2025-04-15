n_instance = 8

def check_number(e, ep, ed, epd, p, pd, d):
    if e + ep + ed + epd + p + pd + d == n_instance:
        return True

def check_valid(e, ep, ed, epd, p, pd, d):
    has_e = e > 0 or ep > 0 or ed > 0 or epd > 0
    has_p = ep > 0 or epd > 0 or p > 0 or pd > 0
    has_d = ed > 0 or epd > 0 or pd > 0 or d > 0
    return has_e and has_p and has_d

combinations = []
for e in range(0, n_instance + 1):
    for ep in range(0, n_instance + 1):
        for ed in range(0, n_instance + 1):
            for epd in range(0, n_instance + 1):
                for p in range(0, n_instance + 1):
                    for pd in range(0, n_instance + 1):
                        for d in range(0, n_instance + 1):
                            if not check_number(e, ep, ed, epd, p, pd, d):
                                continue
                            if not check_valid(e, ep, ed, epd, p, pd, d):
                                continue
                            # if e > 0:
                            #     continue
                            # if epd > 0:
                            #     continue
                            # if pd > 0:
                            #     continue
                            # if ed > 0 and ep > 0:
                            #     continue
                            if ep > 0:
                                continue
                            if ed > 0:
                                continue
                            if epd > 0:
                                continue
                            if pd > 0:
                                continue
                            combinations.append((e, ep, ed, epd, p, pd, d))

# sort based number of engine contained p
combinations.sort(key=lambda x: (x[1] + x[3] + x[4] + x[5]))
for e, ep, ed, epd, p, pd, d in combinations:
    print(f'{e} {ep} {ed} {epd} {p} {pd} {d}')
print('e  ep ed epd p  pd d')
print(f'total {len(combinations)}')